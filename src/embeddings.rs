use crate::{doc_loader::Document, error::ServerError};
use async_openai::{
    Client as OpenAIClient, config::OpenAIConfig, error::ApiError as OpenAIAPIErr,
    types::CreateEmbeddingRequestArgs,
};
use futures::stream::{self, StreamExt};
use ndarray::{Array1, ArrayView1};
use std::sync::Arc;
use std::sync::OnceLock;
use tiktoken_rs::cl100k_base;

pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub struct CachedDocumentEmbedding {
    pub path: String,
    pub content: String,
    pub vector: Vec<f32>,
}

/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    let dot_product = v1.dot(&v2);
    let norm_v1 = v1.dot(&v1).sqrt();
    let norm_v2 = v2.dot(&v2).sqrt();

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        0.0
    } else {
        dot_product / (norm_v1 * norm_v2)
    }
}

/// Generates embeddings for a list of documents using the OpenAI API.
pub async fn generate_embeddings(
    client: &OpenAIClient<OpenAIConfig>,
    documents: &[Document],
    model: &str,
) -> Result<(Vec<(String, Array1<f32>)>, usize), ServerError> {
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    const CONCURRENCY_LIMIT: usize = 8;
    const TOKEN_LIMIT: usize = 8000;

    let documents_vec: Vec<_> = documents.iter().cloned().enumerate().collect();
    let results = stream::iter(documents_vec)
        .map(|(index, doc)| {
            let client = client.clone();
            let model = model.to_string();
            let bpe = Arc::clone(&bpe);

            async move {
                let token_count = bpe.encode_with_special_tokens(&doc.content).len();

                if token_count > TOKEN_LIMIT {
                    return Ok::<Option<(String, Array1<f32>, usize)>, ServerError>(None);
                }

                let inputs: Vec<String> = vec![doc.content.clone()];

                let request = CreateEmbeddingRequestArgs::default()
                    .model(&model)
                    .input(inputs)
                    .build()?;

                let response = client.embeddings().create(request).await?;

                if response.data.len() != 1 {
                    return Err(ServerError::OpenAI(
                        async_openai::error::OpenAIError::ApiError(OpenAIAPIErr {
                            message: format!(
                                "Mismatch in response length for document {}. Expected 1, got {}.",
                                index + 1,
                                response.data.len()
                            ),
                            r#type: Some("sdk_error".to_string()),
                            param: None,
                            code: None,
                        }),
                    ));
                }

                let embedding_data = response.data.first().unwrap();
                let embedding_array = Array1::from(embedding_data.embedding.clone());
                Ok(Some((doc.path.clone(), embedding_array, token_count)))
            }
        })
        .buffer_unordered(CONCURRENCY_LIMIT)
        .collect::<Vec<Result<Option<(String, Array1<f32>, usize)>, ServerError>>>()
        .await;

    let mut embeddings_vec = Vec::new();
    let mut total_processed_tokens: usize = 0;
    for result in results {
        match result {
            Ok(Some((path, embedding, tokens))) => {
                embeddings_vec.push((path, embedding));
                total_processed_tokens += tokens;
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("Error during concurrent embedding generation: {e}");
                return Err(e);
            }
        }
    }

    eprintln!(
        "Finished generating embeddings. Successfully processed {} documents ({} tokens).",
        embeddings_vec.len(),
        total_processed_tokens
    );
    Ok((embeddings_vec, total_processed_tokens))
}
