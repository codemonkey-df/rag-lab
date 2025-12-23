import React, { useState } from "react";
import type { RAGTechnique } from "../services/types";
import { TechniqueSelector } from "./TechniqueSelector";
import { QueryInput } from "./QueryInput";
import { ResultsDisplay } from "./ResultsDisplay";
import { useAppContext } from "../contexts/AppContext";
import { useRAGQuery } from "../hooks/useRAGQuery";

export const SingleView: React.FC = () => {
  const { documentId, sessionId, setSessionId } = useAppContext();
  const { single, executeQuery } = useRAGQuery();

  const [techniques, setTechniques] = useState<RAGTechnique[]>([
    "standard_chunking",
    "basic_rag",
  ]);

  const handleExecuteQuery = async (query: string, params: Record<string, any>) => {
    if (!documentId) {
      alert("Please upload a document first");
      return;
    }

    const request = {
      document_id: documentId,
      query,
      techniques,
      query_params: params,
      session_id: sessionId,
    };

    const result = await executeQuery(request);
    if (result && !sessionId) {
      setSessionId(result.result_id);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Techniques */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <TechniqueSelector
            selected={techniques}
            onChange={setTechniques}
            title="RAG Configuration"
          />
        </div>

        {/* Info Card */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border border-blue-200">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">
            How it Works
          </h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">1.</span>
              <span>Set indexing strategy (Layer 1) when uploading document</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">2.</span>
              <span>Choose retrieval and enhancement techniques (Layer 2)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">3.</span>
              <span>Optionally add orchestration controller (Layer 3)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-blue-600 font-bold">4.</span>
              <span>Enter your query and execute</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Query Input */}
      <div>
        <QueryInput
          onExecute={handleExecuteQuery}
          loading={single.loading}
          buttonLabel="Execute Query"
        />
      </div>

      {/* Results */}
      <ResultsDisplay
        result={single.result}
        loading={single.loading}
        error={single.error}
      />
    </div>
  );
};
