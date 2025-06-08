mod doc_loader;
mod embeddings;
mod error;
mod server;

use crate::{
    embeddings::OPENAI_CLIENT,
    error::ServerError,
    server::{RustDocsServer, load_and_cache_crate_docs},
};
use async_openai::{Client as OpenAIClient, config::OpenAIConfig};
use cargo::core::PackageIdSpec;
use clap::Parser;
use rmcp::{ServiceExt, transport::io::stdio};
use std::env;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The package ID specification (e.g., "serde@^1.0", "tokio").
    #[arg()]
    package_spec: String,

    /// Optional features to enable for the crate when generating documentation.
    #[arg(short = 'F', long, value_delimiter = ',', num_args = 0..)]
    features: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();
    let specid_str = cli.package_spec.trim().to_string();
    let features = cli
        .features
        .map(|f| f.into_iter().map(|s| s.trim().to_string()).collect());

    let spec = PackageIdSpec::parse(&specid_str).map_err(|e| {
        ServerError::Config(format!(
            "Failed to parse package ID spec '{specid_str}': {e}"
        ))
    })?;

    let crate_name = spec.name().to_string();
    let crate_version_req = spec
        .version()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "*".to_string());

    eprintln!(
        "Target Spec: {specid_str}, Parsed Name: {crate_name}, Version Req: {crate_version_req}, Features: {features:?}"
    );

    let _openai_api_key = env::var("OPENAI_API_KEY")
        .map_err(|_| ServerError::MissingEnvVar("OPENAI_API_KEY".to_string()))?;

    let openai_client = if let Ok(api_base) = env::var("OPENAI_API_BASE") {
        let config = OpenAIConfig::new().with_api_base(api_base);
        OpenAIClient::with_config(config)
    } else {
        OpenAIClient::new()
    };
    OPENAI_CLIENT
        .set(openai_client.clone())
        .expect("Failed to set OpenAI client");

    let (
        documents_for_server,
        final_embeddings,
        loaded_from_cache,
        generated_tokens,
        generation_cost,
    ) = load_and_cache_crate_docs(
        &crate_name,
        Some(&crate_version_req),
        features.as_ref(),
        &openai_client,
    )
    .await?;

    eprintln!(
        "Initializing server for crate: {crate_name} (Version Req: {crate_version_req}, Features: {features:?})"
    );

    let features_str = features
        .as_ref()
        .map(|f| format!(" Features: {f:?}"))
        .unwrap_or_default();

    let startup_message = if loaded_from_cache {
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Loaded {} embeddings from cache.",
            crate_name,
            crate_version_req,
            features_str,
            final_embeddings.len()
        )
    } else {
        let tokens = generated_tokens.unwrap_or(0);
        let cost = generation_cost.unwrap_or(0.0);
        format!(
            "Server for crate '{}' (Version Req: '{}'{}) initialized. Generated {} embeddings for {} tokens (Est. Cost: ${:.6}).",
            crate_name,
            crate_version_req,
            features_str,
            final_embeddings.len(),
            tokens,
            cost
        )
    };

    let service = RustDocsServer::new(
        crate_name.clone(),
        documents_for_server,
        final_embeddings,
        startup_message,
    )?;

    eprintln!("Rust Docs MCP server starting via stdio...");

    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {e:?}");
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("{} Docs MCP server running...", &crate_name);

    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {e:?}");
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("Rust Docs MCP server stopped.");
    Ok(())
}
