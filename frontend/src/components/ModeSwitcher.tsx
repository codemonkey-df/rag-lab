import React from "react";
import type { AppMode } from "../services/types";
import { useAppContext } from "../contexts/AppContext";

export const ModeSwitcher: React.FC = () => {
  const { mode, setMode, reset } = useAppContext();

  const handleModeChange = (newMode: AppMode) => {
    if (newMode !== mode) {
      // Only reset session-related state, preserve the document
      // The document should persist across mode switches
      setMode(newMode);
    }
  };

  return (
    <div className="flex items-center justify-between bg-gradient-to-r from-slate-900 to-slate-800 text-white px-6 py-4 shadow-lg">
      <div className="flex items-center gap-8">
        <h1 className="text-2xl font-bold">RAG Lab Playground</h1>
        
        <div className="flex gap-4">
          <button
            onClick={() => handleModeChange("single")}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              mode === "single"
                ? "bg-blue-600 text-white shadow-lg"
                : "bg-slate-700 hover:bg-slate-600 text-gray-200"
            }`}
          >
            Single Query
          </button>
          
          <button
            onClick={() => handleModeChange("comparison")}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              mode === "comparison"
                ? "bg-blue-600 text-white shadow-lg"
                : "bg-slate-700 hover:bg-slate-600 text-gray-200"
            }`}
          >
            Comparison
          </button>
        </div>
      </div>

      <button
        onClick={() => {
          reset();
          window.location.reload();
        }}
        className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors text-sm"
      >
        Reset All
      </button>
    </div>
  );
};
