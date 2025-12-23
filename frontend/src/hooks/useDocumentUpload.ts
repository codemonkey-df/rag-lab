import { useState } from "react";
import { api } from "../services/api";
import type { Document } from "../services/types";

interface UploadState {
  loading: boolean;
  error: string | null;
  document: Document | null;
}

export const useDocumentUpload = () => {
  const [state, setState] = useState<UploadState>({
    loading: false,
    error: null,
    document: null,
  });

  const uploadDocument = async (
    file: File,
    chunking_strategy: string,
    chunk_size: number = 1024,
    chunk_overlap: number = 200,
    session_id?: string
  ): Promise<Document | null> => {
    setState({ loading: true, error: null, document: null });

    try {
      const response = await api.uploadDocument(
        file,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
        session_id
      );

      // Fetch the document details
      const doc = await api.getDocument(response.document_id);
      setState({ loading: false, error: null, document: doc });
      return doc;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Upload failed";
      setState({ loading: false, error: errorMessage, document: null });
      return null;
    }
  };

  const pollDocument = async (doc_id: string): Promise<Document | null> => {
    try {
      const doc = await api.getDocument(doc_id);
      setState((prev) => ({ ...prev, document: doc }));
      return doc;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch document";
      setState((prev) => ({ ...prev, error: errorMessage }));
      return null;
    }
  };

  return {
    ...state,
    uploadDocument,
    pollDocument,
  };
};
