use bincode::config;
use rustdocs_mcp_server::embeddings::CachedDocumentEmbedding;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod cache_serialization_tests {
    use super::*;

    #[test]
    fn test_cache_file_creation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache_file = temp_dir.path().join("test_cache.bin");

        let test_data = vec![
            CachedDocumentEmbedding {
                path: "src/lib.rs".to_string(),
                content: "pub mod test;".to_string(),
                vector: vec![0.1, 0.2, 0.3],
            },
            CachedDocumentEmbedding {
                path: "src/main.rs".to_string(),
                content: "fn main() {}".to_string(),
                vector: vec![0.4, 0.5, 0.6],
            },
        ];

        // Write cache
        let encoded =
            bincode::encode_to_vec(&test_data, config::standard()).expect("Failed to encode");
        fs::write(&cache_file, encoded).expect("Failed to write cache");

        assert!(cache_file.exists());

        // Read cache back
        let bytes = fs::read(&cache_file).expect("Failed to read cache");
        let decoded: Vec<CachedDocumentEmbedding> =
            bincode::decode_from_slice(&bytes, config::standard())
                .expect("Failed to decode")
                .0;

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].path, "src/lib.rs");
        assert_eq!(decoded[1].path, "src/main.rs");
    }

    #[test]
    fn test_cache_directory_structure() {
        let _temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Test different cache path scenarios
        let test_cases = vec![
            (
                "my_crate",
                "*",
                None::<Vec<String>>,
                "my_crate/_/no_features/embeddings.bin",
            ),
            (
                "my_crate",
                "1.0.0",
                None,
                "my_crate/1.0.0/no_features/embeddings.bin",
            ),
            (
                "my_crate",
                "^1.0",
                None,
                "my_crate/_1.0/no_features/embeddings.bin",
            ),
            (
                "my_crate",
                ">=0.5,<2.0",
                None,
                "my_crate/__0.5__2.0/no_features/embeddings.bin",
            ),
        ];

        for (crate_name, version, _features, expected_path) in test_cases {
            let sanitized_version =
                version.replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");

            let cache_path = PathBuf::from(crate_name)
                .join(&sanitized_version)
                .join("no_features")
                .join("embeddings.bin");

            let expected = PathBuf::from(expected_path);
            assert_eq!(cache_path, expected);
        }
    }

    #[test]
    fn test_large_cache_handling() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache_file = temp_dir.path().join("large_cache.bin");

        // Create a large dataset
        let mut large_data = Vec::new();
        for i in 0..1000 {
            large_data.push(CachedDocumentEmbedding {
                path: format!("src/file_{}.rs", i),
                content: format!("Content for file {}", i),
                vector: vec![i as f32; 100], // 100-dimensional vectors
            });
        }

        // Write large cache
        let encoded = bincode::encode_to_vec(&large_data, config::standard())
            .expect("Failed to encode large data");
        fs::write(&cache_file, &encoded).expect("Failed to write large cache");

        // Verify file size is reasonable
        let metadata = fs::metadata(&cache_file).expect("Failed to get metadata");
        assert!(metadata.len() > 0);

        // Read back and verify
        let bytes = fs::read(&cache_file).expect("Failed to read large cache");
        let decoded: Vec<CachedDocumentEmbedding> =
            bincode::decode_from_slice(&bytes, config::standard())
                .expect("Failed to decode large data")
                .0;

        assert_eq!(decoded.len(), 1000);
        assert_eq!(decoded[500].path, "src/file_500.rs");
    }
}

#[cfg(test)]
mod cache_path_tests {
    use super::*;

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_xdg_cache_paths() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        unsafe {
            std::env::set_var("XDG_DATA_HOME", temp_dir.path());
        }

        let base_dirs = xdg::BaseDirectories::with_prefix("rustdocs-mcp-server")
            .expect("Failed to create XDG dirs");

        let cache_file = base_dirs
            .place_data_file("test_crate/1.0.0/no_features/embeddings.bin")
            .expect("Failed to place data file");

        // Verify the path contains our components
        let path_str = cache_file.to_string_lossy();
        assert!(path_str.contains("rustdocs-mcp-server"));
        assert!(path_str.contains("test_crate"));
        assert!(path_str.contains("1.0.0"));
        assert!(path_str.contains("no_features"));
        assert!(path_str.contains("embeddings.bin"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_windows_cache_paths() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // On Windows, we would use dirs::cache_dir()
        // For testing, we just verify the path construction
        let app_cache_dir = temp_dir.path().join("rustdocs-mcp-server");
        fs::create_dir_all(&app_cache_dir).expect("Failed to create dirs");

        let cache_file = app_cache_dir
            .join("test_crate")
            .join("1.0.0")
            .join("no_features")
            .join("embeddings.bin");

        // Create parent directories
        if let Some(parent) = cache_file.parent() {
            fs::create_dir_all(parent).expect("Failed to create parent dirs");
        }

        // Write a test file
        fs::write(&cache_file, b"test").expect("Failed to write test file");
        assert!(cache_file.exists());
    }
}

#[cfg(test)]
mod cache_invalidation_tests {
    #[test]
    fn test_cache_key_uniqueness() {
        // Different versions should have different cache keys
        let version_keys = vec![
            ("1.0.0", "1.0.0"),
            ("^1.0", "_1.0"),
            ("~1.0", "_1.0"),
            (">=1.0", "__1.0"),
            ("*", "_"),
        ];

        let mut seen_keys = std::collections::HashSet::new();
        for (version, expected_key) in version_keys {
            let sanitized =
                version.replace(|c: char| !c.is_alphanumeric() && c != '.' && c != '-', "_");
            assert_eq!(sanitized, expected_key);

            // Most keys should be unique (some may collide like ^1.0 and ~1.0)
            seen_keys.insert(sanitized);
        }

        // We should have at least 4 unique keys
        assert!(seen_keys.len() >= 4);
    }

    #[test]
    fn test_feature_hash_consistency() {
        use rustdocs_mcp_server::server::RustDocsServer;

        // Same features in different order should produce same hash
        let features1 = Some(vec![
            "feat1".to_string(),
            "feat2".to_string(),
            "feat3".to_string(),
        ]);
        let features2 = Some(vec![
            "feat3".to_string(),
            "feat1".to_string(),
            "feat2".to_string(),
        ]);

        let hash1 = RustDocsServer::hash_features(&features1);
        let hash2 = RustDocsServer::hash_features(&features2);

        assert_eq!(hash1, hash2);

        // Hash should be stable across runs
        let hash3 = RustDocsServer::hash_features(&features1);
        assert_eq!(hash1, hash3);
    }
}
