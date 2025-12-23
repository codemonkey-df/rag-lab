import React, { useState } from "react";
import type { Document } from "../services/types";
import { useDocumentUpload } from "../hooks/useDocumentUpload";
import { useAppContext } from "../contexts/AppContext";
import { useRAGConfiguration } from "../hooks/useRAGConfiguration";
import { api } from "../services/api";

interface FileUploadProps {
  onUploadComplete?: (doc: Document) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete }) => {
  const { uploadDocument, loading, error, document: hookDocument } = useDocumentUpload();
  const { sessionId, setSessionId, setDocumentId, setDocument, document: contextDocument } = useAppContext();
  const { config: ragConfig } = useRAGConfiguration();
  
  // Use context document as source of truth, fallback to hook document during upload
  const uploadedDoc = contextDocument || hookDocument;
  
  const [dragOver, setDragOver] = useState(false);
  const [chunking_strategy, setChunkingStrategy] = useState(
    ragConfig?.defaults.indexing_strategy || "standard"
  );
  const [chunk_size, setChunkSize] = useState(
    ragConfig?.defaults.chunk_size || 1024
  );
  const [chunk_overlap, setChunkOverlap] = useState(
    ragConfig?.defaults.chunk_overlap || 200
  );
  const [strategies, setStrategies] = useState<string[]>([]);
  const [strategiesInfo, setStrategiesInfo] = useState<Record<string, { label?: string; description?: string; warning?: string }>>({});
  const [pollInterval, setPollInterval] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);

  // Update defaults when config loads
  React.useEffect(() => {
    if (ragConfig?.defaults) {
      setChunkingStrategy(ragConfig.defaults.indexing_strategy);
      setChunkSize(ragConfig.defaults.chunk_size);
      setChunkOverlap(ragConfig.defaults.chunk_overlap);
    }
  }, [ragConfig?.defaults]);

  React.useEffect(() => {
    // Fetch available strategies and their info on mount
    const fetchStrategies = async () => {
      try {
        const data = await api.getAvailableStrategies();
        const strategyNames = Object.keys(data.strategies);
        setStrategies(strategyNames);
        
        // Also get RAG config for labels, descriptions and warnings
        if (ragConfig?.techniques?.layer_1) {
          const layer1Info: Record<string, { label?: string; description?: string; warning?: string }> = {};
          ragConfig.techniques.layer_1.forEach((tech) => {
            // Map technique value to strategy name
            // "standard_chunking" -> "standard"
            // "semantic_chunking" -> "semantic"
            // "parent_document" -> "parent_document"
            // "contextual_headers" -> "headers"
            // "proposition_chunking" -> "proposition"
            let strategyName = tech.value;
            if (strategyName === "contextual_headers") {
              strategyName = "headers";
            } else if (strategyName.endsWith("_chunking")) {
              strategyName = strategyName.replace("_chunking", "");
            }
            layer1Info[strategyName] = {
              label: tech.label,
              description: tech.description,
              warning: tech.warning,
            };
          });
          setStrategiesInfo(layer1Info);
        }
      } catch (err) {
        console.error("Failed to fetch strategies:", err);
      }
    };
    fetchStrategies();
  }, [ragConfig]);

  React.useEffect(() => {
    // Poll document status if upload is in progress
    // Only poll if we have a context document (not just hook document)
    const docToPoll = contextDocument || hookDocument;
    if (docToPoll && docToPoll.status !== "completed" && docToPoll.status !== "failed") {
      const interval = setInterval(async () => {
        try {
          const updated = await api.getDocument(docToPoll.id);
          setDocument(updated);
          setProgress(updated.indexing_progress || 0);

          if (updated.status === "completed") {
            clearInterval(interval);
            setPollInterval(null);
          }
        } catch (err) {
          console.error("Failed to poll document:", err);
        }
      }, 2000);

      setPollInterval(interval);
      return () => clearInterval(interval);
    } else if (docToPoll?.status === "completed") {
      clearInterval(pollInterval || undefined);
      setPollInterval(null);
    }
  }, [contextDocument, hookDocument]);

  const handleFileDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      alert("Please select a PDF file");
      return;
    }

    try {
      const doc = await uploadDocument(
        file,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
        sessionId
      );

      if (doc) {
        if (!sessionId) {
          // Extract session ID from the response if available
          setSessionId(doc.session_id);
        }
        setDocumentId(doc.id);
        setDocument(doc);
        onUploadComplete?.(doc);
      }
    } catch (err) {
      console.error("Upload failed:", err);
    }
  };

  // Only show document info if we have a context document (not just hook state)
  if (contextDocument && contextDocument.status !== "failed") {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Document</h2>
          {contextDocument.status === "completed" && (
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              Ready
            </span>
          )}
          {contextDocument.status === "processing" && (
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
              Indexing...
            </span>
          )}
        </div>

        <div className="space-y-2 text-gray-700">
          <p>
            <span className="font-semibold">File:</span> {contextDocument.filename}
          </p>
          <p>
            <span className="font-semibold">Strategy:</span> {contextDocument.chunking_strategy}
          </p>
          <p>
            <span className="font-semibold">Chunk Size:</span> {contextDocument.chunk_size}
          </p>
          <p>
            <span className="font-semibold">Chunk Overlap:</span> {contextDocument.chunk_overlap}
          </p>
        </div>

        {contextDocument.status === "processing" && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-600 mt-2">{progress}% complete</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-4">Upload Document</h2>

      <div className="space-y-4 mb-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Indexing Strategy
          </label>
          <select
            value={chunking_strategy}
            onChange={(e) => setChunkingStrategy(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
          >
            {strategies.map((strat) => {
              const info = strategiesInfo[strat];
              const displayLabel = info?.label || strat.replace(/_/g, " ").split(" ").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
              const shortDesc = info?.description ? ` - ${info.description}` : "";
              return (
                <option key={strat} value={strat}>
                  {displayLabel}{shortDesc}
                </option>
              );
            })}
          </select>
          
          {/* Strategy Description and Warning - Matching Layer 1 style */}
          {strategiesInfo[chunking_strategy] && (
            <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg space-y-2">
              {strategiesInfo[chunking_strategy].label && (
                <p className="text-sm font-medium text-gray-700">
                  {strategiesInfo[chunking_strategy].label}
                </p>
              )}
              {strategiesInfo[chunking_strategy].description && (
                <p className="text-xs text-gray-600">
                  {strategiesInfo[chunking_strategy].description}
                </p>
              )}
              {strategiesInfo[chunking_strategy].warning && (
                <div className="flex items-start gap-1 pt-1">
                  <span className="text-orange-600 text-xs">⚠️</span>
                  <p className="text-xs text-orange-700 font-medium">
                    {strategiesInfo[chunking_strategy].warning}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chunk Size
            </label>
            <input
              type="number"
              value={chunk_size}
              onChange={(e) => setChunkSize(parseInt(e.target.value))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chunk Overlap
            </label>
            <input
              type="number"
              value={chunk_overlap}
              onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 bg-white"
            />
          </div>
        </div>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleFileDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-gray-50 hover:border-gray-400"
        }`}
      >
        <svg
          className="mx-auto h-12 w-12 text-gray-400 mb-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>

        <p className="text-gray-600 mb-2">Drag and drop your PDF here</p>
        <p className="text-sm text-gray-500 mb-4">or</p>

        <label className="inline-block">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            disabled={loading}
            className="hidden"
          />
          <span className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer font-medium transition-colors inline-block disabled:opacity-50">
            {loading ? "Uploading..." : "Select PDF"}
          </span>
        </label>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}
    </div>
  );
};
