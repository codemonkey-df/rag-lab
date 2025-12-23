import React, { useState } from "react";
import type { ChunkInfo } from "../services/types";

interface ChunkDisplayProps {
  chunks: ChunkInfo[];
  title?: string;
}

export const ChunkDisplay: React.FC<ChunkDisplayProps> = ({
  chunks,
  title = "Retrieved Chunks",
}) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (!chunks || chunks.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
        <p className="text-gray-500">No chunks retrieved</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        {title} ({chunks.length})
      </h3>

      <div className="space-y-3">
        {chunks.map((chunk, index) => (
          <div
            key={index}
            className="border border-gray-200 rounded-lg overflow-hidden hover:border-gray-300 transition-colors"
          >
            <button
              onClick={() =>
                setExpandedIndex(expandedIndex === index ? null : index)
              }
              className="w-full px-4 py-3 hover:bg-gray-50 flex items-start justify-between"
            >
              <div className="flex-1 text-left">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-medium text-gray-700">
                    Page {chunk.page}
                  </span>
                  {chunk.score !== undefined && (
                    <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                      Score: {(chunk.score * 100).toFixed(1)}%
                    </span>
                  )}
                  {chunk.line_start !== undefined && (
                    <span className="text-xs text-gray-600">
                      Lines {chunk.line_start}-{chunk.line_end}
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 line-clamp-2">
                  {chunk.text}
                </p>
              </div>
              <span
                className={`ml-2 transition-transform flex-shrink-0 ${
                  expandedIndex === index ? "rotate-180" : ""
                }`}
              >
                ▼
              </span>
            </button>

            {expandedIndex === index && (
              <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {chunk.text}
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
