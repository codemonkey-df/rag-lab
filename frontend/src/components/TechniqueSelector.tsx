import React, { useState, useEffect } from "react";
import type { RAGTechnique } from "../services/types";
import { useRAGConfiguration } from "../hooks/useRAGConfiguration";
import { useAppContext } from "../contexts/AppContext";
import { api } from "../services/api";

interface TechniqueSelectorProps {
  selected: RAGTechnique[];
  onChange: (techniques: RAGTechnique[]) => void;
  title?: string;
}

const getTechniqueName = (technique: string): string => {
  return technique
    .replace(/_/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

export const TechniqueSelector: React.FC<TechniqueSelectorProps> = ({
  selected,
  onChange,
  title = "RAG Techniques",
}) => {
  const { config } = useRAGConfiguration();
  const { document } = useAppContext();
  const [openLayers, setOpenLayers] = useState<Set<number>>(new Set([2])); // Open Layer 2 by default
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  
  // Hide Layer 1 if document is already uploaded (indexing strategy is set during upload)
  const hideLayer1 = !!document && document.status !== "failed";
  
  // Validate techniques whenever selection changes
  // Include document's indexing strategy (Layer 1) if available
  useEffect(() => {
    const validateSelection = async () => {
      if (selected.length === 0 && !document?.chunking_strategy) {
        setValidationErrors([]);
        setValidationWarnings([]);
        return;
      }
      
      try {
        // Build techniques array including Layer 1 from document if available
        let techniquesToValidate = [...selected];
        
        // Map document's chunking_strategy to Layer 1 technique if document exists
        if (document?.chunking_strategy) {
          const strategyToTechnique: Record<string, string> = {
            "standard": "standard_chunking",
            "semantic": "semantic_chunking",
            "parent_document": "parent_document",
            "headers": "contextual_headers",
            "proposition": "proposition_chunking",
          };
          
          const layer1Technique = strategyToTechnique[document.chunking_strategy];
          if (layer1Technique && !techniquesToValidate.includes(layer1Technique as RAGTechnique)) {
            techniquesToValidate.push(layer1Technique as RAGTechnique);
          }
        }
        
        const result = await api.validateTechniques(techniquesToValidate);
        setValidationErrors(result.errors || []);
        setValidationWarnings(result.warnings || []);
      } catch (error) {
        console.error("Validation error:", error);
        // Don't show validation errors if API call fails
        setValidationErrors([]);
        setValidationWarnings([]);
      }
    };
    
    validateSelection();
  }, [selected, document]);

  const toggleLayer = (layer: number) => {
    const newOpen = new Set(openLayers);
    if (newOpen.has(layer)) {
      newOpen.delete(layer);
    } else {
      newOpen.add(layer);
    }
    setOpenLayers(newOpen);
  };

  if (!config) {
    return <div className="text-gray-500">Loading configuration...</div>;
  }

  const handleSelect = (technique: string) => {
    const layer1Techs = config.techniques.layer_1.map((t) => t.value);
    const layer3Techs = config.techniques.layer_3.map((t) => t.value);

    let newSelected = [...selected];

    if (layer1Techs.includes(technique)) {
      // Layer 1: mutually exclusive - remove other Layer 1 techniques
      newSelected = newSelected.filter((t) => !layer1Techs.includes(t));
      if (!newSelected.includes(technique as RAGTechnique)) {
        newSelected.push(technique as RAGTechnique);
      }
    } else if (layer3Techs.includes(technique)) {
      // Layer 3: at most one - remove other Layer 3 techniques
      // Allow unselecting if already selected
      if (newSelected.includes(technique as RAGTechnique)) {
        newSelected = newSelected.filter((t) => t !== technique);
      } else {
        newSelected = newSelected.filter((t) => !layer3Techs.includes(t));
        newSelected.push(technique as RAGTechnique);
      }
    } else {
      // Layer 2: multi-select
      if (newSelected.includes(technique as RAGTechnique)) {
        newSelected = newSelected.filter((t) => t !== technique);
      } else {
        newSelected.push(technique as RAGTechnique);
      }
    }

    onChange(newSelected);
  };

  const renderTechniqueGroup = (
    groupNum: number,
    label: string,
    techniques: typeof config.techniques.layer_1,
    helpText: string
  ) => {
    const isOpen = openLayers.has(groupNum);

    return (
      <div key={groupNum} className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          onClick={() => toggleLayer(groupNum)}
          className="w-full px-4 py-3 bg-gray-100 hover:bg-gray-150 flex items-center justify-between font-semibold text-gray-700"
        >
          <span>{label}</span>
          <span className={`transition-transform ${isOpen ? "rotate-180" : ""}`}>
            ▼
          </span>
        </button>

        {isOpen && (
          <div className="p-4 bg-white space-y-2">
            <p className="text-xs text-gray-600 mb-3">{helpText}</p>
            <div className="space-y-2">
              {techniques.map((techniqueInfo) => (
                <div key={techniqueInfo.value} className="mb-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={selected.includes(techniqueInfo.value as RAGTechnique)}
                      onChange={() => handleSelect(techniqueInfo.value)}
                      className="rounded border-gray-300 text-blue-600 cursor-pointer"
                    />
                    <span className="ml-2 text-gray-700 font-medium">
                      {techniqueInfo.label}
                    </span>
                  </label>
                  <p className="text-xs text-gray-500 ml-6 mt-1">
                    {techniqueInfo.description}
                  </p>
                  {techniqueInfo.warning && (
                    <p className="text-xs text-orange-600 ml-6 mt-1">
                      ⚠️ {techniqueInfo.warning}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>

      {!hideLayer1 && renderTechniqueGroup(
        1,
        "Layer 1: Indexing Strategy",
        config.techniques.layer_1,
        "Choose ONE indexing strategy (required)"
      )}

      {hideLayer1 && (
        <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-semibold text-gray-700">
              Layer 1: Indexing Strategy
            </span>
            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
              Set during upload
            </span>
          </div>
          <p className="text-xs text-gray-600">
            Indexing strategy is configured when uploading the document. To change it, upload a new document.
          </p>
          {document && (
            <p className="text-xs text-gray-700 mt-2 font-medium">
              Current: <span className="font-semibold">{document.chunking_strategy.replace(/_/g, " ").toUpperCase()}</span>
            </p>
          )}
        </div>
      )}

      {renderTechniqueGroup(
        2,
        "Layer 2: Pipeline Components",
        config.techniques.layer_2,
        "Combine multiple retrieval, expansion, and filtering techniques"
      )}

      {renderTechniqueGroup(
        3,
        "Layer 3: Orchestration",
        config.techniques.layer_3,
        "Choose at most ONE advanced orchestration controller"
      )}

      {!hideLayer1 && selected.length === 0 && (
        <p className="text-sm text-red-600">
          ⚠️ At least select an indexing strategy (Layer 1)
        </p>
      )}

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm font-semibold text-red-900 mb-2">Validation Errors:</p>
          <ul className="list-disc list-inside space-y-1">
            {validationErrors.map((error, idx) => (
              <li key={idx} className="text-sm text-red-700">{error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Validation Warnings */}
      {validationWarnings.length > 0 && (
        <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
          <p className="text-sm font-semibold text-orange-900 mb-2">Warnings:</p>
          <ul className="list-disc list-inside space-y-1">
            {validationWarnings.map((warning, idx) => (
              <li key={idx} className="text-sm text-orange-700">{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {selected.length > 0 && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-900">
            <span className="font-semibold">Selected:</span>{" "}
            {selected.map((t) => getTechniqueName(t)).join(", ")}
          </p>
        </div>
      )}
    </div>
  );
};
