use ndarray::Array1;
use rustdocs_mcp_server::{
    doc_loader::Document,
    embeddings::{CachedDocumentEmbedding, cosine_similarity},
};

#[cfg(test)]
mod cosine_similarity_tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let v2 = Array1::from(vec![1.0, 0.0, 0.0]);

        let similarity = cosine_similarity(v1.view(), v2.view());
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let v1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let v2 = Array1::from(vec![0.0, 1.0, 0.0]);

        let similarity = cosine_similarity(v1.view(), v2.view());
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let v1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let v2 = Array1::from(vec![-1.0, 0.0, 0.0]);

        let similarity = cosine_similarity(v1.view(), v2.view());
        assert!((similarity - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let v1 = Array1::from(vec![0.0, 0.0, 0.0]);
        let v2 = Array1::from(vec![1.0, 0.0, 0.0]);

        let similarity = cosine_similarity(v1.view(), v2.view());
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_normalized_vectors() {
        let v1 = Array1::from(vec![0.6, 0.8, 0.0]);
        let v2 = Array1::from(vec![0.8, 0.6, 0.0]);

        let similarity = cosine_similarity(v1.view(), v2.view());
        // 0.6*0.8 + 0.8*0.6 = 0.48 + 0.48 = 0.96
        assert!((similarity - 0.96).abs() < 1e-6);
    }
}

#[cfg(test)]
mod cached_document_embedding_tests {
    use super::*;

    #[test]
    fn test_cached_document_embedding_creation() {
        let embedding = CachedDocumentEmbedding {
            path: "test/path".to_string(),
            content: "Test content".to_string(),
            vector: vec![1.0, 2.0, 3.0],
        };

        assert_eq!(embedding.path, "test/path");
        assert_eq!(embedding.content, "Test content");
        assert_eq!(embedding.vector, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cached_document_embedding_serialization() {
        use bincode::config;

        let embedding = CachedDocumentEmbedding {
            path: "test/doc.rs".to_string(),
            content: "/// Documentation for test module".to_string(),
            vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        };

        // Test encoding
        let encoded =
            bincode::encode_to_vec(&embedding, config::standard()).expect("Failed to encode");
        assert!(!encoded.is_empty());

        // Test decoding
        let decoded: CachedDocumentEmbedding =
            bincode::decode_from_slice(&encoded, config::standard())
                .expect("Failed to decode")
                .0;

        assert_eq!(decoded.path, embedding.path);
        assert_eq!(decoded.content, embedding.content);
        assert_eq!(decoded.vector, embedding.vector);
    }

    #[test]
    fn test_cached_document_embedding_vector_conversion() {
        let embedding = CachedDocumentEmbedding {
            path: "test/path".to_string(),
            content: "Test".to_string(),
            vector: vec![1.0, 2.0, 3.0],
        };

        // Test conversion to Array1
        let array = Array1::from(embedding.vector.clone());
        assert_eq!(array.len(), 3);
        assert_eq!(array[0], 1.0);
        assert_eq!(array[1], 2.0);
        assert_eq!(array[2], 3.0);
    }
}

#[cfg(test)]
mod embedding_generation_tests {
    use super::*;

    #[test]
    fn test_document_to_embedding_mapping() {
        let documents = vec![
            Document {
                path: "src/lib.rs".to_string(),
                content: "pub mod test;".to_string(),
            },
            Document {
                path: "src/test.rs".to_string(),
                content: "fn test() {}".to_string(),
            },
        ];

        // Simulate embedding results
        let embeddings = vec![
            ("src/lib.rs".to_string(), Array1::<f32>::zeros(10)),
            ("src/test.rs".to_string(), Array1::<f32>::ones(10)),
        ];

        // Verify that paths match
        for (i, doc) in documents.iter().enumerate() {
            assert_eq!(doc.path, embeddings[i].0);
        }
    }
}
