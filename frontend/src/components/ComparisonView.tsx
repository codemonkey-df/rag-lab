import React, { useState } from "react";
import type { RAGTechnique } from "../services/types";
import { TechniqueSelector } from "./TechniqueSelector";
import { QueryInput } from "./QueryInput";
import { ResultsDisplay } from "./ResultsDisplay";
import { useAppContext } from "../contexts/AppContext";
import { useRAGQuery } from "../hooks/useRAGQuery";

export const ComparisonView: React.FC = () => {
  const { documentId, sessionId, setSessionId } = useAppContext();
  const { comparison, compareQueries } = useRAGQuery();
  
  const [techniques1, setTechniques1] = useState<RAGTechnique[]>([
    "standard_chunking",
    "basic_rag",
  ]);
  const [techniques2, setTechniques2] = useState<RAGTechnique[]>([
    "standard_chunking",
    "fusion_retrieval",
  ]);

  const handleExecuteComparison = async (q: string, params: Record<string, any>) => {
    if (!documentId) {
      alert("Please upload a document first");
      return;
    }

    const request = {
      document_id: documentId,
      query: q,
      pipeline_1: {
        techniques: techniques1,
        query_params: params,
      },
      pipeline_2: {
        techniques: techniques2,
        query_params: params,
      },
      session_id: sessionId,
    };

    const result = await compareQueries(request);
    if (result && !sessionId) {
      setSessionId(result.pipeline_1_result.result_id);
    }
  };

  const getInterpretationColor = (interpretation: string): string => {
    if (interpretation.includes("Very similar")) return "text-green-600";
    if (interpretation.includes("Similar")) return "text-blue-600";
    if (interpretation.includes("Different") && !interpretation.includes("Very"))
      return "text-orange-600";
    return "text-red-600";
  };

  const getSimilarityBarColor = (similarity: number): string => {
    if (similarity > 0.8) return "bg-green-500";
    if (similarity > 0.6) return "bg-blue-500";
    if (similarity > 0.4) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-100 to-slate-50 rounded-lg p-6 border border-slate-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Pipeline Comparison
        </h2>
        <p className="text-gray-600">
          Configure and compare two different RAG pipelines with the same query
        </p>
      </div>

      {/* Configuration and Query Input */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Pipeline */}
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-blue-900">Pipeline 1</h3>
          </div>
          <TechniqueSelector
            selected={techniques1}
            onChange={setTechniques1}
            title="Pipeline 1 Techniques"
          />
        </div>

        {/* Right Pipeline */}
        <div className="space-y-4">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-purple-900">Pipeline 2</h3>
          </div>
          <TechniqueSelector
            selected={techniques2}
            onChange={setTechniques2}
            title="Pipeline 2 Techniques"
          />
        </div>
      </div>

      {/* Shared Query Input */}
      <div>
        <div className="mb-2 text-sm font-medium text-gray-700">Shared Query</div>
        <QueryInput
          onExecute={handleExecuteComparison}
          loading={comparison.loading}
          buttonLabel="Compare Pipelines"
        />
      </div>

      {/* Results */}
      {comparison.result && (
        <div className="space-y-6">
          {/* Comparison Metrics */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">
              Comparison Results
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Similarity Score */}
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border border-blue-200">
                <p className="text-sm font-medium text-blue-900 mb-2">
                  Semantic Similarity
                </p>
                <div className="flex items-end gap-4">
                  <div className="flex-1">
                    <p className="text-4xl font-bold text-blue-600 mb-2">
                      {(comparison.result.comparison.semantic_similarity * 100).toFixed(1)}%
                    </p>
                    <div className="w-full bg-blue-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${getSimilarityBarColor(
                          comparison.result.comparison.semantic_similarity
                        )}`}
                        style={{
                          width: `${
                            comparison.result.comparison.semantic_similarity * 100
                          }%`,
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Interpretation */}
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6 border border-purple-200">
                <p className="text-sm font-medium text-purple-900 mb-2">
                  Interpretation
                </p>
                <p
                  className={`text-2xl font-bold ${getInterpretationColor(
                    comparison.result.comparison.interpretation
                  )}`}
                >
                  {comparison.result.comparison.interpretation}
                </p>
              </div>

              {/* Latency Difference */}
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-6 border border-orange-200">
                <p className="text-sm font-medium text-orange-900 mb-2">
                  Latency Difference
                </p>
                <p className="text-3xl font-bold text-orange-600">
                  {comparison.result.comparison.latency_diff_ms.toFixed(0)}ms
                </p>
              </div>
            </div>
          </div>

          {/* Side-by-Side Results */}
          <div className="grid grid-cols-2 gap-6">
            {/* Pipeline 1 Results */}
            <div className="border-2 border-blue-200 rounded-lg p-6 bg-blue-50">
              <h3 className="text-lg font-semibold text-blue-900 mb-4">
                Pipeline 1 Results
              </h3>
              <div className="space-y-4">
                {/* Latency */}
                <div className="bg-white rounded p-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">Latency</span>
                  <span className="font-semibold text-blue-600">
                    {comparison.result.pipeline_1_result.scores.latency_ms.toFixed(0)}ms
                  </span>
                </div>

                {/* Response */}
                <div className="bg-white rounded p-3">
                  <p className="text-xs font-semibold text-gray-600 mb-2">Response</p>
                  <p className="text-sm text-gray-700 line-clamp-4">
                    {comparison.result.pipeline_1_result.response}
                  </p>
                </div>

                {/* Chunks Count */}
                <div className="bg-white rounded p-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">
                    Chunks Retrieved
                  </span>
                  <span className="font-semibold text-blue-600">
                    {comparison.result.pipeline_1_result.retrieved_chunks.length}
                  </span>
                </div>
              </div>

              {/* Full results */}
              <div className="mt-4 max-h-96 overflow-y-auto">
                <ResultsDisplay result={comparison.result.pipeline_1_result} />
              </div>
            </div>

            {/* Pipeline 2 Results */}
            <div className="border-2 border-purple-200 rounded-lg p-6 bg-purple-50">
              <h3 className="text-lg font-semibold text-purple-900 mb-4">
                Pipeline 2 Results
              </h3>
              <div className="space-y-4">
                {/* Latency */}
                <div className="bg-white rounded p-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">Latency</span>
                  <span className="font-semibold text-purple-600">
                    {comparison.result.pipeline_2_result.scores.latency_ms.toFixed(0)}ms
                  </span>
                </div>

                {/* Response */}
                <div className="bg-white rounded p-3">
                  <p className="text-xs font-semibold text-gray-600 mb-2">Response</p>
                  <p className="text-sm text-gray-700 line-clamp-4">
                    {comparison.result.pipeline_2_result.response}
                  </p>
                </div>

                {/* Chunks Count */}
                <div className="bg-white rounded p-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">
                    Chunks Retrieved
                  </span>
                  <span className="font-semibold text-purple-600">
                    {comparison.result.pipeline_2_result.retrieved_chunks.length}
                  </span>
                </div>
              </div>

              {/* Full results */}
              <div className="mt-4 max-h-96 overflow-y-auto">
                <ResultsDisplay result={comparison.result.pipeline_2_result} />
              </div>
            </div>
          </div>
        </div>
      )}

      {comparison.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error</h3>
          <p className="text-red-700">{comparison.error}</p>
        </div>
      )}

      {comparison.loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <span className="ml-4 text-gray-600 text-lg">Comparing pipelines...</span>
        </div>
      )}
    </div>
  );
};
