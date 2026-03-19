flowchart TB

    %% --- ROW 1 ---
    subgraph Pipeline Processing
        direction LR

        subgraph P1[Phase 1 - Training]
            direction TB
            DS[MVTec AD Dataset]
            PREP[Preprocessing]
            TRAIN[Autoencoder Training]
            CAL[Calibration]
            MODEL[Checkpoint]
            DS --> PREP --> TRAIN --> CAL --> MODEL
        end

        subgraph P2[Phase 2 - Streaming]
            direction TB
            SIM[Frame Simulator]
            DBX[Databricks + Spark]
            SCORE[Anomaly Scoring]
            DELTA[Event Log]
            SIM --> SCORE
            DBX --> SCORE
            SCORE --> DELTA
        end
    end

    %% --- ROW 2 ---
    subgraph LLM
        direction LR

        subgraph P3[Phase 3 - LLM Diagnostics]
            direction TB
            API[FastAPI]
            VLM[VLM]
            FAISS[Vector DB]
            GRAPH[LangGraph]
            TICKET[Ticket Draft]
            API --> VLM --> GRAPH --> TICKET
            FAISS --> GRAPH
        end

        subgraph P4[        Application]
            direction TB
            MONGO[(MongoDB)]
            DASH[Dashboard]
            MONGO --> DASH
        end
    end

    %% --- CROSS CONNECTIONS ---
    MODEL --> SCORE
    DELTA --> API
    API --> DASH
    TICKET --> MONGO

    %% --- STYLES ---
    classDef dataset fill:#dbeafe,stroke:#1d4ed8;
    classDef ml fill:#dcfce7,stroke:#16a34a;
    classDef stream fill:#ede9fe,stroke:#7c3aed;
    classDef orchestration fill:#fef3c7,stroke:#d97706;
    classDef app fill:#fee2e2,stroke:#dc2626;

    class DS dataset;
    class PREP,TRAIN,CAL,MODEL ml;
    class SIM,DBX,SCORE,DELTA stream;
    class API,VLM,FAISS,GRAPH,TICKET orchestration;
    class MONGO,DASH app;