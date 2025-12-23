import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { RAGConfiguration } from "../services/types";

interface UseRAGConfigurationState {
  config: RAGConfiguration | null;
  loading: boolean;
  error: string | null;
}

export const useRAGConfiguration = () => {
  const [state, setState] = useState<UseRAGConfigurationState>({
    config: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    const fetchConfiguration = async () => {
      try {
        const config = await api.getRAGConfiguration();
        setState({ config, loading: false, error: null });
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Failed to fetch RAG configuration";
        setState({ config: null, loading: false, error: errorMessage });
      }
    };

    fetchConfiguration();
  }, []);

  return state;
};
