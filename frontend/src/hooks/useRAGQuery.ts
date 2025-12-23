import { useState } from "react";
import { api } from "../services/api";
import type {
  QueryRequest,
  QueryResponse,
  ComparisonRequest,
  ComparisonResponse,
} from "../services/types";

interface SingleQueryState {
  loading: boolean;
  error: string | null;
  result: QueryResponse | null;
}

interface ComparisonState {
  loading: boolean;
  error: string | null;
  result: ComparisonResponse | null;
}

export const useRAGQuery = () => {
  const [singleState, setSingleState] = useState<SingleQueryState>({
    loading: false,
    error: null,
    result: null,
  });

  const [comparisonState, setComparisonState] = useState<ComparisonState>({
    loading: false,
    error: null,
    result: null,
  });

  const executeQuery = async (request: QueryRequest): Promise<QueryResponse | null> => {
    setSingleState({ loading: true, error: null, result: null });

    try {
      const result = await api.executeQuery(request);
      setSingleState({ loading: false, error: null, result });
      return result;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Query execution failed";
      setSingleState({ loading: false, error: errorMessage, result: null });
      return null;
    }
  };

  const compareQueries = async (
    request: ComparisonRequest
  ): Promise<ComparisonResponse | null> => {
    setComparisonState({ loading: true, error: null, result: null });

    try {
      const result = await api.comparePipelines(request);
      setComparisonState({ loading: false, error: null, result });
      return result;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Comparison failed";
      setComparisonState({ loading: false, error: errorMessage, result: null });
      return null;
    }
  };

  return {
    single: singleState,
    comparison: comparisonState,
    executeQuery,
    compareQueries,
  };
};
