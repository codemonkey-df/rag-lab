import React, { useState } from "react";

interface QueryInputProps {
  onExecute: (query: string, params: Record<string, any>) => void;
  loading?: boolean;
  buttonLabel?: string;
}

export const QueryInput: React.FC<QueryInputProps> = ({
  onExecute,
  loading = false,
  buttonLabel = "Execute Query",
}) => {
  const [query, setQuery] = useState("");
  const [top_k, setTopK] = useState(5);
  const [bm25_weight, setBm25Weight] = useState(0.5);
  const [temperature, setTemperature] = useState(0.7);

  const handleSubmit = () => {
    if (!query.trim()) {
      alert("Please enter a query");
      return;
    }

    onExecute(query, {
      top_k,
      bm25_weight,
      temperature,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.ctrlKey && e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Your Query
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your document... (Ctrl+Enter to submit)"
          className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-24 resize-none text-gray-900 bg-white"
        />
        <p className="text-xs text-gray-500 mt-1">
          Tip: Press Ctrl+Enter to execute
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Top K
            <span className="text-gray-500 font-normal"> ({top_k})</span>
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={top_k}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            BM25 Weight
            <span className="text-gray-500 font-normal">
              {" "}
              ({bm25_weight.toFixed(2)})
            </span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={bm25_weight}
            onChange={(e) => setBm25Weight(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Temperature
            <span className="text-gray-500 font-normal">
              {" "}
              ({temperature.toFixed(2)})
            </span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading || !query.trim()}
        className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-semibold transition-colors"
      >
        {loading ? "Executing..." : buttonLabel}
      </button>
    </div>
  );
};
