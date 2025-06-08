use async_openai::{Client as OpenAIClient, config::OpenAIConfig};
use cargo::core::PackageIdSpec;
use rustdocs_mcp_server::{error::ServerError, server::load_and_cache_crate_docs};

/// Create a mock OpenAI client that will fail
fn create_failing_openai_client() -> OpenAIClient<OpenAIConfig> {
    // Point to a non-existent endpoint to simulate network failure
    let config = OpenAIConfig::new()
        .with_api_base("http://localhost:1")
        .with_api_key("invalid_key");
    OpenAIClient::with_config(config)
}

#[cfg(test)]
mod invalid_crate_tests {
    use super::*;

    #[test]
    fn test_invalid_package_spec_parsing() {
        let invalid_specs = vec![
            "@",
            "@@version",
            "crate@@@",
            "crate name with spaces@1.0",
            "",
        ];

        for spec in invalid_specs {
            let result = PackageIdSpec::parse(spec);
            assert!(result.is_err(), "Expected error for spec: {}", spec);
        }
    }

    #[test]
    fn test_valid_package_spec_parsing() {
        // PackageIdSpec is for exact package identification, not version requirements
        // It expects formats like: "name:version" or just "name"
        let valid_specs = vec![
            ("serde", "serde", None),
            ("serde:1.0.0", "serde", Some("1.0.0")),
            ("async-trait:0.1.74", "async-trait", Some("0.1.74")),
            ("tokio:1.34.0", "tokio", Some("1.34.0")),
        ];

        for (spec, expected_name, expected_version) in valid_specs {
            let result = PackageIdSpec::parse(spec);
            assert!(result.is_ok(), "Expected success for spec: {}", spec);

            let parsed = result.unwrap();
            assert_eq!(parsed.name(), expected_name);

            if let Some(expected_ver) = expected_version {
                assert!(
                    parsed.version().is_some(),
                    "Expected version for spec: {}",
                    spec
                );
                assert_eq!(parsed.version().unwrap().to_string(), expected_ver);
            } else {
                assert!(parsed.version().is_none());
            }
        }
    }

    #[tokio::test]
    async fn test_nonexistent_crate() {
        let client = create_failing_openai_client();
        let crate_name = "this-crate-definitely-does-not-exist-12345";
        let version = Some("1.0.0");
        let features = None;

        let result =
            load_and_cache_crate_docs(crate_name, version, features.as_ref(), &client).await;

        // This should fail during document loading
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod feature_handling_tests {
    #[test]
    fn test_feature_list_handling() {
        let feature_lists = vec![
            (
                Some(vec!["tokio-runtime".to_string(), "async".to_string()]),
                Some(vec!["async".to_string(), "tokio-runtime".to_string()]),
                true, // Should produce same hash
            ),
            (
                Some(vec!["feature1".to_string()]),
                Some(vec!["feature2".to_string()]),
                false, // Should produce different hash
            ),
            (
                Some(vec![]),
                None,
                false, // Empty vec vs None should be different
            ),
        ];

        for (features1, features2, should_match) in feature_lists {
            let hash1 = rustdocs_mcp_server::server::RustDocsServer::hash_features(&features1);
            let hash2 = rustdocs_mcp_server::server::RustDocsServer::hash_features(&features2);

            if should_match {
                assert_eq!(
                    hash1, hash2,
                    "Hashes should match for {:?} and {:?}",
                    features1, features2
                );
            } else {
                assert_ne!(
                    hash1, hash2,
                    "Hashes should differ for {:?} and {:?}",
                    features1, features2
                );
            }
        }
    }
}

#[cfg(test)]
mod error_propagation_tests {
    use super::*;

    #[test]
    fn test_server_error_conversions() {
        // Test that various error types can be converted to ServerError
        use std::io;

        // IO Error
        let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let server_error: ServerError = io_error.into();
        assert!(matches!(server_error, ServerError::Io(_)));

        // JSON Error
        let json_str = "{ invalid json }";
        let json_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let server_error: ServerError = json_error.into();
        assert!(matches!(server_error, ServerError::Json(_)));
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            ServerError::MissingEnvVar("TEST_VAR".to_string()),
            ServerError::Config("Invalid configuration".to_string()),
            ServerError::Tiktoken("Token error".to_string()),
            ServerError::Xdg("XDG directory error".to_string()),
            ServerError::McpRuntime("Runtime error".to_string()),
        ];

        for error in errors {
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
        }
    }
}

#[cfg(test)]
mod cache_corruption_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_corrupted_cache_handling() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        unsafe {
            std::env::set_var("XDG_DATA_HOME", temp_dir.path());
        }

        // Create a corrupted cache file
        let cache_path = temp_dir
            .path()
            .join("rustdocs-mcp-server")
            .join("test_crate")
            .join("_")
            .join("no_features");

        fs::create_dir_all(&cache_path).expect("Failed to create cache dirs");

        let cache_file = cache_path.join("embeddings.bin");
        fs::write(&cache_file, b"corrupted data that is not valid bincode")
            .expect("Failed to write corrupted cache");

        // Attempt to load with corrupted cache should regenerate
        let client = create_failing_openai_client();
        let result = load_and_cache_crate_docs("test_crate", None, None, &client).await;

        // This will fail due to the failing client, but it should handle the corrupted cache gracefully
        assert!(result.is_err());
    }
}
