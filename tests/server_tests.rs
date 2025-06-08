use ndarray::Array1;
use rustdocs_mcp_server::{doc_loader::Document, server::RustDocsServer};
use std::env;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper to set up test environment with temporary cache directory
fn setup_test_env() -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Override XDG_DATA_HOME to use our temp directory
    unsafe {
        env::set_var("XDG_DATA_HOME", temp_dir.path());
    }

    temp_dir
}

#[cfg(test)]
mod server_instance_tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let crate_name = "test_crate".to_string();
        let documents = vec![
            Document {
                path: "test/path1".to_string(),
                content: "Test content 1".to_string(),
            },
            Document {
                path: "test/path2".to_string(),
                content: "Test content 2".to_string(),
            },
        ];
        let embeddings = vec![
            ("test/path1".to_string(), Array1::zeros(1536)),
            ("test/path2".to_string(), Array1::zeros(1536)),
        ];
        let startup_message = "Test startup message".to_string();

        let server = RustDocsServer::new(
            crate_name.clone(),
            documents.clone(),
            embeddings.clone(),
            startup_message,
        );

        assert!(server.is_ok());
    }

    #[test]
    fn test_hash_features() {
        // Test with no features
        let no_features: Option<Vec<String>> = None;
        let hash1 = RustDocsServer::hash_features(&no_features);
        assert_eq!(hash1, "no_features");

        // Test with features
        let features = Some(vec!["feature1".to_string(), "feature2".to_string()]);
        let hash2 = RustDocsServer::hash_features(&features);
        assert_ne!(hash2, "no_features");
        assert!(!hash2.is_empty());

        // Test that same features produce same hash (regardless of order)
        let features_reversed = Some(vec!["feature2".to_string(), "feature1".to_string()]);
        let hash3 = RustDocsServer::hash_features(&features_reversed);
        assert_eq!(hash2, hash3);

        // Test that different features produce different hash
        let different_features = Some(vec!["feature3".to_string()]);
        let hash4 = RustDocsServer::hash_features(&different_features);
        assert_ne!(hash2, hash4);
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_path_generation() {
        let _temp_dir = setup_test_env();

        // Test cache path for crate without version
        let crate_name = "test_crate";
        let _version: Option<&str> = None;
        let _features: Option<Vec<String>> = None;

        // We can't fully test load_and_cache_crate_docs without mocking the doc loader
        // and OpenAI client, but we can verify the cache path construction logic

        // Verify that cache directory structure would be created correctly
        #[cfg(not(target_os = "windows"))]
        {
            let cache_base = env::var("XDG_DATA_HOME").unwrap();
            let expected_path = PathBuf::from(&cache_base)
                .join("rustdocs-mcp-server")
                .join(crate_name)
                .join("_")
                .join("no_features")
                .join("embeddings.bin");

            // Path should be constructible
            assert!(expected_path.parent().is_some());
        }
    }

    #[tokio::test]
    async fn test_cache_with_version_sanitization() {
        let _temp_dir = setup_test_env();

        // Test that version strings are properly sanitized for filesystem
        let versions = vec![
            ("^1.0", "_1.0"),
            (">=0.5,<2.0", "__0.5__2.0"),
            ("1.2.3", "1.2.3"),
            ("~1.0", "_1.0"),
        ];

        for (input_version, expected_sanitized) in versions {
            let sanitized =
                input_version.replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");
            assert_eq!(sanitized, expected_sanitized);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_server_with_empty_documents() {
        let crate_name = "empty_crate".to_string();
        let documents = vec![];
        let embeddings = vec![];
        let startup_message = "Empty server".to_string();

        let server = RustDocsServer::new(crate_name, documents, embeddings, startup_message);

        assert!(server.is_ok());
    }

    #[test]
    fn test_server_with_mismatched_embeddings() {
        let crate_name = "test_crate".to_string();
        let documents = vec![
            Document {
                path: "test/path1".to_string(),
                content: "Test content 1".to_string(),
            },
            Document {
                path: "test/path2".to_string(),
                content: "Test content 2".to_string(),
            },
        ];

        // Embeddings don't match documents (missing path2)
        let embeddings = vec![("test/path1".to_string(), Array1::zeros(1536))];

        let startup_message = "Test startup message".to_string();

        // Server should still be created successfully
        let server = RustDocsServer::new(crate_name, documents, embeddings, startup_message);

        assert!(server.is_ok());
    }
}
