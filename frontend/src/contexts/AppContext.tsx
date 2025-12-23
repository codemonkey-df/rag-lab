import React, { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";
import type { AppMode, Document, AppContextType } from "../services/types";

const AppContext = createContext<AppContextType | undefined>(undefined);

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [mode, setMode] = useState<AppMode>("single");
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [documentId, setDocumentId] = useState<string | undefined>();
  const [document, setDocument] = useState<Document | undefined>();

  const reset = () => {
    setMode("single");
    setSessionId(undefined);
    setDocumentId(undefined);
    setDocument(undefined);
  };

  const value: AppContextType = {
    mode,
    sessionId,
    documentId,
    document,
    setMode,
    setSessionId,
    setDocumentId,
    setDocument,
    reset,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
};
