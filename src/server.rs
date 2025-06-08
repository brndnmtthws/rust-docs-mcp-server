use crate::{
    doc_loader::Document,
    embeddings::{CachedDocumentEmbedding, OPENAI_CLIENT, cosine_similarity, generate_embeddings},
    error::ServerError,
};
use async_openai::Client as OpenAIClient;
use async_openai::types::{
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
};
use bincode::config;
use ndarray::Array1;
use rmcp::model::AnnotateAble;
use rmcp::{
    Error as McpError, Peer, ServerHandler,
    model::{
        CallToolResult, Content, GetPromptRequestParam, GetPromptResult, Implementation,
        ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult, LoggingLevel,
        LoggingMessageNotification, LoggingMessageNotificationMethod,
        LoggingMessageNotificationParam, Notification, PaginatedRequestParam, ProtocolVersion,
        RawResource, ReadResourceRequestParam, ReadResourceResult, Resource, ResourceContents,
        ServerCapabilities, ServerInfo, ServerNotification,
    },
    service::{RequestContext, RoleServer},
    tool,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::hash_map::DefaultHasher,
    env,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::BufReader,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::Mutex;
#[cfg(not(target_os = "windows"))]
use xdg::BaseDirectories;

#[derive(Debug, Deserialize, JsonSchema)]
struct QueryRustDocsArgs {
    #[schemars(description = "The specific question about the crate's API or usage.")]
    question: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FetchCrateDocsArgs {
    #[schemars(description = "The name of the Rust crate to fetch documentation for.")]
    crate_name: String,
    #[schemars(
        description = "The version requirement for the crate (e.g., '^1.0', '>=0.5,<2.0'). Defaults to '*' (latest)."
    )]
    version: Option<String>,
    #[schemars(description = "Features to enable when generating documentation.")]
    features: Option<Vec<String>>,
}

#[derive(Clone)]
pub struct RustDocsServer {
    crate_name: Arc<String>,
    documents: Arc<Vec<Document>>,
    embeddings: Arc<Vec<(String, Array1<f32>)>>,
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
    startup_message: Arc<Mutex<Option<String>>>,
    startup_message_sent: Arc<Mutex<bool>>,
}

pub async fn load_and_cache_crate_docs(
    crate_name: &str,
    version: Option<&str>,
    features: Option<&Vec<String>>,
    openai_client: &OpenAIClient<async_openai::config::OpenAIConfig>,
) -> Result<
    (
        Vec<Document>,
        Vec<(String, Array1<f32>)>,
        bool,
        Option<usize>,
        Option<f64>,
    ),
    ServerError,
> {
    let crate_version_req = version.unwrap_or("*");

    let sanitized_version_req =
        crate_version_req.replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");
    let features_hash = RustDocsServer::hash_features(&features.cloned());

    let embeddings_relative_path = PathBuf::from(crate_name)
        .join(&sanitized_version_req)
        .join(&features_hash)
        .join("embeddings.bin");

    #[cfg(not(target_os = "windows"))]
    let embeddings_file_path = {
        let xdg_dirs = BaseDirectories::with_prefix("rustdocs-mcp-server")
            .map_err(|e| ServerError::Xdg(format!("Failed to get XDG directories: {e}")))?;
        xdg_dirs
            .place_data_file(embeddings_relative_path)
            .map_err(ServerError::Io)?
    };

    #[cfg(target_os = "windows")]
    let embeddings_file_path = {
        let cache_dir = dirs::cache_dir().ok_or_else(|| {
            ServerError::Config("Could not determine cache directory on Windows".to_string())
        })?;
        let app_cache_dir = cache_dir.join("rustdocs-mcp-server");
        fs::create_dir_all(&app_cache_dir).map_err(ServerError::Io)?;
        app_cache_dir.join(embeddings_relative_path)
    };

    eprintln!("Cache file path: {embeddings_file_path:?}");

    let mut loaded_from_cache = false;
    let mut loaded_embeddings: Option<Vec<(String, Array1<f32>)>> = None;
    let mut loaded_documents_from_cache: Option<Vec<Document>> = None;

    if embeddings_file_path.exists() {
        eprintln!("Attempting to load cached data from: {embeddings_file_path:?}");
        match File::open(&embeddings_file_path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                match bincode::decode_from_reader::<Vec<CachedDocumentEmbedding>, _, _>(
                    reader,
                    config::standard(),
                ) {
                    Ok(cached_data) => {
                        eprintln!(
                            "Successfully loaded {} items from cache. Separating data...",
                            cached_data.len()
                        );
                        let mut embeddings = Vec::with_capacity(cached_data.len());
                        let mut documents = Vec::with_capacity(cached_data.len());
                        for item in cached_data {
                            embeddings.push((item.path.clone(), Array1::from(item.vector)));
                            documents.push(Document {
                                path: item.path,
                                content: item.content,
                            });
                        }
                        loaded_embeddings = Some(embeddings);
                        loaded_documents_from_cache = Some(documents);
                        loaded_from_cache = true;
                    }
                    Err(e) => {
                        eprintln!("Failed to decode cache file: {e}. Will regenerate.");
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to open cache file: {e}. Will regenerate.");
            }
        }
    } else {
        eprintln!("Cache file not found. Will generate.");
    }

    let mut generated_tokens: Option<usize> = None;
    let mut generation_cost: Option<f64> = None;
    let documents_for_server: Vec<Document>;

    let final_embeddings = match loaded_embeddings {
        Some(embeddings) => {
            eprintln!("Using embeddings and documents loaded from cache.");
            documents_for_server = loaded_documents_from_cache.unwrap();
            embeddings
        }
        None => {
            eprintln!("Proceeding with documentation loading and embedding generation.");

            eprintln!(
                "Loading documents for crate: {crate_name} (Version Req: {crate_version_req}, Features: {features:?})"
            );

            let loaded_documents = crate::doc_loader::load_documents(
                crate_name,
                crate_version_req,
                features.cloned().as_ref(),
            )?;
            eprintln!("Loaded {} documents.", loaded_documents.len());
            documents_for_server = loaded_documents.clone();

            eprintln!("Generating embeddings...");
            let embedding_model: String = env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            let (generated_embeddings, total_tokens) =
                generate_embeddings(openai_client, &loaded_documents, &embedding_model).await?;

            let cost_per_million = 0.02;
            let estimated_cost = (total_tokens as f64 / 1_000_000.0) * cost_per_million;
            eprintln!("Embedding generation cost for {total_tokens} tokens: ${estimated_cost:.6}");
            generated_tokens = Some(total_tokens);
            generation_cost = Some(estimated_cost);

            eprintln!("Saving generated documents and embeddings to: {embeddings_file_path:?}");

            let mut combined_cache_data: Vec<CachedDocumentEmbedding> = Vec::new();
            let embedding_map: std::collections::HashMap<String, Array1<f32>> =
                generated_embeddings.clone().into_iter().collect();

            for doc in &loaded_documents {
                if let Some(embedding_array) = embedding_map.get(&doc.path) {
                    combined_cache_data.push(CachedDocumentEmbedding {
                        path: doc.path.clone(),
                        content: doc.content.clone(),
                        vector: embedding_array.to_vec(),
                    });
                } else {
                    eprintln!(
                        "Warning: Embedding not found for document path: {}. Skipping from cache.",
                        doc.path
                    );
                }
            }

            match bincode::encode_to_vec(&combined_cache_data, config::standard()) {
                Ok(encoded_bytes) => {
                    #[allow(clippy::collapsible_if)]
                    if let Some(parent_dir) = embeddings_file_path.parent() {
                        if !parent_dir.exists() {
                            if let Err(e) = fs::create_dir_all(parent_dir) {
                                eprintln!(
                                    "Warning: Failed to create cache directory {}: {e}",
                                    parent_dir.display()
                                );
                            }
                        }
                    }
                    if let Err(e) = fs::write(&embeddings_file_path, encoded_bytes) {
                        eprintln!("Warning: Failed to write cache file: {e}");
                    } else {
                        eprintln!(
                            "Cache saved successfully ({} items).",
                            combined_cache_data.len()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to encode data for cache: {e}");
                }
            }
            generated_embeddings
        }
    };

    Ok((
        documents_for_server,
        final_embeddings,
        loaded_from_cache,
        generated_tokens,
        generation_cost,
    ))
}

impl RustDocsServer {
    pub fn new(
        crate_name: String,
        documents: Vec<Document>,
        embeddings: Vec<(String, Array1<f32>)>,
        startup_message: String,
    ) -> Result<Self, ServerError> {
        Ok(Self {
            crate_name: Arc::new(crate_name),
            documents: Arc::new(documents),
            embeddings: Arc::new(embeddings),
            peer: Arc::new(Mutex::new(None)),
            startup_message: Arc::new(Mutex::new(Some(startup_message))),
            startup_message_sent: Arc::new(Mutex::new(false)),
        })
    }

    pub fn send_log(&self, level: LoggingLevel, message: String) {
        let peer_arc = Arc::clone(&self.peer);
        tokio::spawn(async move {
            let mut peer_guard = peer_arc.lock().await;
            if let Some(peer) = peer_guard.as_mut() {
                let params = LoggingMessageNotificationParam {
                    level,
                    logger: None,
                    data: serde_json::Value::String(message),
                };
                let log_notification: LoggingMessageNotification = Notification {
                    method: LoggingMessageNotificationMethod,
                    params,
                };
                let server_notification =
                    ServerNotification::LoggingMessageNotification(log_notification);
                if let Err(e) = peer.send_notification(server_notification).await {
                    eprintln!("Failed to send MCP log notification: {e}");
                }
            } else {
                eprintln!("Log task ran but MCP peer was not connected.");
            }
        });
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    pub fn hash_features(features: &Option<Vec<String>>) -> String {
        features
            .as_ref()
            .map(|f| {
                let mut sorted_features = f.clone();
                sorted_features.sort_unstable();
                let mut hasher = DefaultHasher::new();
                sorted_features.hash(&mut hasher);
                format!("{:x}", hasher.finish())
            })
            .unwrap_or_else(|| "no_features".to_string())
    }
}

#[tool(tool_box)]
impl RustDocsServer {
    #[tool(
        description = "Query documentation for a specific Rust crate using semantic search and LLM summarization."
    )]
    async fn query_rust_docs(
        &self,
        #[tool(aggr)] args: QueryRustDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        let mut sent_guard = self.startup_message_sent.lock().await;
        if !*sent_guard {
            let mut msg_guard = self.startup_message.lock().await;
            if let Some(message) = msg_guard.take() {
                self.send_log(LoggingLevel::Info, message);
                *sent_guard = true;
            }
            drop(msg_guard);
            drop(sent_guard);
        } else {
            drop(sent_guard);
        }

        let question = &args.question;

        self.send_log(
            LoggingLevel::Info,
            format!(
                "Received query for crate '{}': {}",
                self.crate_name, question
            ),
        );

        let client = OPENAI_CLIENT
            .get()
            .ok_or_else(|| McpError::internal_error("OpenAI client not initialized", None))?;

        let embedding_model: String =
            env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());
        let question_embedding_request = CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(question.to_string())
            .build()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to build embedding request: {e}"), None)
            })?;

        let question_embedding_response = client
            .embeddings()
            .create(question_embedding_request)
            .await
            .map_err(|e| McpError::internal_error(format!("OpenAI API error: {e}"), None))?;

        let question_embedding = question_embedding_response.data.first().ok_or_else(|| {
            McpError::internal_error("Failed to get embedding for question", None)
        })?;

        let question_vector = Array1::from(question_embedding.embedding.clone());

        let mut best_match: Option<(&str, f32)> = None;
        for (path, doc_embedding) in self.embeddings.iter() {
            let score = cosine_similarity(question_vector.view(), doc_embedding.view());
            if best_match.is_none() || score > best_match.unwrap().1 {
                best_match = Some((path, score));
            }
        }

        let response_text = match best_match {
            Some((best_path, _score)) => {
                eprintln!("Best match found: {best_path}");
                let context_doc = self.documents.iter().find(|doc| doc.path == best_path);

                if let Some(doc) = context_doc {
                    let system_prompt = format!(
                        "You are an expert technical assistant for the Rust crate '{}'. \
                         Answer the user's question based *only* on the provided context. \
                         If the context does not contain the answer, say so. \
                         Do not make up information. Be clear, concise, and comprehensive providing example usage code when possible.",
                        self.crate_name
                    );
                    let user_prompt = format!(
                        "Context:\n---\n{}\n---\n\nQuestion: {}",
                        doc.content, question
                    );

                    let llm_model: String = env::var("LLM_MODEL")
                        .unwrap_or_else(|_| "gpt-4o-mini-2024-07-18".to_string());
                    let chat_request = CreateChatCompletionRequestArgs::default()
                        .model(llm_model)
                        .messages(vec![
                            ChatCompletionRequestSystemMessageArgs::default()
                                .content(system_prompt)
                                .build()
                                .map_err(|e| {
                                    McpError::internal_error(
                                        format!("Failed to build system message: {e}"),
                                        None,
                                    )
                                })?
                                .into(),
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(user_prompt)
                                .build()
                                .map_err(|e| {
                                    McpError::internal_error(
                                        format!("Failed to build user message: {e}"),
                                        None,
                                    )
                                })?
                                .into(),
                        ])
                        .build()
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to build chat request: {e}"),
                                None,
                            )
                        })?;

                    let chat_response = client.chat().create(chat_request).await.map_err(|e| {
                        McpError::internal_error(format!("OpenAI chat API error: {e}"), None)
                    })?;

                    chat_response
                        .choices
                        .first()
                        .and_then(|choice| choice.message.content.clone())
                        .unwrap_or_else(|| "Error: No response from LLM.".to_string())
                } else {
                    "Error: Could not find content for best matching document.".to_string()
                }
            }
            None => "Could not find any relevant document context.".to_string(),
        };

        Ok(CallToolResult::success(vec![Content::text(format!(
            "From {} docs: {}",
            self.crate_name, response_text
        ))]))
    }

    #[tool(
        description = "Fetch and cache documentation for any Rust crate. Returns information about the loaded documentation including whether it was loaded from cache."
    )]
    async fn fetch_crate_docs(
        &self,
        #[tool(aggr)] args: FetchCrateDocsArgs,
    ) -> Result<CallToolResult, McpError> {
        let crate_name = &args.crate_name;
        let version = args.version.as_deref();
        let features = args.features.as_ref();

        self.send_log(
            LoggingLevel::Info,
            format!(
                "Fetching documentation for crate '{crate_name}' (version: {version:?}, features: {features:?})"
            ),
        );

        let client = OPENAI_CLIENT
            .get()
            .ok_or_else(|| McpError::internal_error("OpenAI client not initialized", None))?;

        match load_and_cache_crate_docs(crate_name, version, features, client).await {
            Ok((documents, embeddings, from_cache, tokens, cost)) => {
                let status_message = if from_cache {
                    format!(
                        "Successfully loaded {} documents and {} embeddings from cache for crate '{}'.",
                        documents.len(),
                        embeddings.len(),
                        crate_name
                    )
                } else {
                    let tokens_info = tokens
                        .map(|t| format!(" Generated {t} tokens"))
                        .unwrap_or_default();
                    let cost_info = cost
                        .map(|c| format!(" (estimated cost: ${c:.6})"))
                        .unwrap_or_default();
                    format!(
                        "Successfully loaded {} documents and generated {} embeddings for crate '{}'.{}{}",
                        documents.len(),
                        embeddings.len(),
                        crate_name,
                        tokens_info,
                        cost_info
                    )
                };

                self.send_log(LoggingLevel::Info, status_message.clone());

                let result_data = json!({
                    "crate_name": crate_name,
                    "version": version.unwrap_or("*"),
                    "features": features,
                    "documents_count": documents.len(),
                    "embeddings_count": embeddings.len(),
                    "loaded_from_cache": from_cache,
                    "tokens_generated": tokens,
                    "estimated_cost": cost,
                    "status": "success",
                    "message": status_message
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&result_data).unwrap_or(status_message),
                )]))
            }
            Err(e) => {
                let error_message =
                    format!("Failed to fetch documentation for crate '{crate_name}': {e}");

                self.send_log(LoggingLevel::Error, error_message.clone());

                Err(McpError::internal_error(error_message, None))
            }
        }
    }
}

#[tool(tool_box)]
impl ServerHandler for RustDocsServer {
    fn get_info(&self) -> ServerInfo {
        let capabilities = ServerCapabilities::builder()
            .enable_tools()
            .enable_logging()
            .build();

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities,
            server_info: Implementation {
                name: "rust-docs-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(format!(
                "This server provides tools to work with Rust crate documentation. \
                 The server was initialized for the '{}' crate, but you can: \
                 1. Use 'query_rust_docs' to ask questions about the '{}' crate \
                 2. Use 'fetch_crate_docs' to load documentation for any other Rust crate \
                 All documentation is cached for efficient reuse.",
                self.crate_name, self.crate_name
            )),
        }
    }

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text(&format!("crate://{}", self.crate_name), "crate_name"),
            ],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let expected_uri = format!("crate://{}", self.crate_name);
        if request.uri == expected_uri {
            Ok(ReadResourceResult {
                contents: vec![ResourceContents::text(
                    self.crate_name.as_str(),
                    &request.uri,
                )],
            })
        } else {
            Err(McpError::resource_not_found(
                format!("Resource URI not found: {}", request.uri),
                Some(json!({ "uri": request.uri })),
            ))
        }
    }

    async fn list_prompts(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult {
            next_cursor: None,
            prompts: Vec::new(),
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        Err(McpError::invalid_params(
            format!("Prompt not found: {}", request.name),
            None,
        ))
    }

    async fn list_resource_templates(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
        })
    }
}
