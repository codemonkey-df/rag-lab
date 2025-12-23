import React, { useEffect } from "react";
import { ModeSwitcher } from "./components/ModeSwitcher";
import { FileUpload } from "./components/FileUpload";
import { SingleView } from "./components/SingleView";
import { ComparisonView } from "./components/ComparisonView";
import { useAppContext } from "./contexts/AppContext";
import { api } from "./services/api";

function App() {
  const { mode, document: uploadedDoc } = useAppContext();
  const [isHealthy, setIsHealthy] = React.useState(true);
  const [healthError, setHealthError] = React.useState<string | null>(null);

  useEffect(() => {
    // Check backend health on mount
    const checkHealth = async () => {
      try {
        const health = await api.healthCheck();
        setIsHealthy(health.healthy);
        if (!health.healthy) {
          setHealthError(health.message);
        } else {
          setHealthError(null);
        }
      } catch (err: any) {
        setIsHealthy(false);
        const errorMessage = err?.response?.status
          ? `Backend returned error ${err.response.status}: ${err.response.statusText}`
          : err?.message?.includes("timeout")
          ? "Backend request timed out. Is the server running?"
          : err?.message?.includes("Network Error") || err?.code === "ERR_NETWORK"
          ? "Cannot connect to backend. Make sure the RAG Lab backend is running on http://localhost:8000 and CORS is configured."
          : `Connection error: ${err?.message || "Unknown error"}`;
        setHealthError(errorMessage);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <ModeSwitcher />

      {!isHealthy && (
        <div className="bg-red-50 border-b border-red-200 px-6 py-4">
          <div className="max-w-7xl mx-auto">
            <h3 className="text-red-900 font-semibold mb-1">Backend Connection Error</h3>
            <p className="text-red-700">{healthError}</p>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Step 1: File Upload */}
        <div className="mb-8">
          <FileUpload />
        </div>

        {/* Step 2 & 3: Configuration and Results */}
        {uploadedDoc && (
          <>
            {mode === "single" && <SingleView />}
            {mode === "comparison" && <ComparisonView />}
          </>
        )}

        {!uploadedDoc && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 text-center">
            <svg
              className="mx-auto h-12 w-12 text-blue-600 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6v6m0 0v6m0-6h6m0 0h6m0-6v6m0 0v6M3 12a9 9 0 1118 0 9 9 0 01-18 0z"
              />
            </svg>
            <h2 className="text-xl font-semibold text-blue-900 mb-2">
              Ready to start?
            </h2>
            <p className="text-blue-800">
              Upload a PDF file above to begin {mode === "single" ? "querying" : "comparing RAG pipelines"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
