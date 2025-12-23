import React from "react";
import type { QueryResponse } from "../services/types";
import { ChunkDisplay } from "./ChunkDisplay";

interface ResultsDisplayProps {
  result: QueryResponse | null;
  loading?: boolean;
  error?: string | null;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  result,
  loading = false,
  error = null,
}) => {
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-red-900 mb-2">Error</h3>
        <p className="text-red-700">{error}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Executing query...</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center text-gray-500">
        <p>Execute a query to see results here</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Response */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Response</h3>
        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {result.response}
        </p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow-md p-4">
          <p className="text-sm text-gray-600">Latency</p>
          <p className="text-2xl font-bold text-blue-600">
            {result.scores.latency_ms.toFixed(0)}ms
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4">
          <p className="text-sm text-gray-600">Token Count (Est.)</p>
          <p className="text-2xl font-bold text-blue-600">
            {result.scores.token_count_est}
          </p>
        </div>
      </div>

      {/* Chunks */}
      <ChunkDisplay chunks={result.retrieved_chunks} />
    </div>
  );
};
