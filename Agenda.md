## **2025-03-01 Agenda**

# **First Meeting Agenda: Commodity Price Forecasting App** 

## **1. Project Scope & Technical Requirements (20 min)**

### **Target Commodities and Forecasting Horizons**

-   Primary commodities to support (energy, metals, agricultural)

    -   Start with Cocoa or gold

-   Temporal forecasting ranges (intraday, weekly, monthly, yearly)

    -   Start with daily

### **Core Features and API Requirements**

-   Historical price visualization with interactive charting (frontend)

    -   Embed plotly? Streamlit?, Dash

    -   Choose Streamlit

-   Multiple forecasting model options (statistical, ML-based)

    -   Start with Prophet

-   Market insights and correlation analysis

    -   Plot temperature next to commodity prices (Meteostat)

-   Required third-party data APIs and integration points

    -   Meteostat

    -   Api Ninja

-   Internal API structure and design philosophy

### **Performance Expectations**

-   Maximum acceptable latency for data retrieval

    -   Results returned in less than 1 second

-   Real-time update requirements

    -   Once per day

-   Forecast calculation time constraints

    -   30 day bound

-   Scalability targets (number of concurrent users)

    -   100

-   Mobile vs. desktop performance considerations

    -   Start with Webapp

### **Technical Constraints**

-   Budget limitations for infrastructure and data services

-   Security and compliance requirements

    -   Oauth

-   Timeline constraints for MVP and subsequent releases

    -   MVP (29th March)

## **2. System Architecture Discussion (25 min)**

### **Backend Framework Options**

-   Comparison of Django vs. Flask vs. Node.js vs. Spring Boot

    -   Flask

-   Discussion of language tradeoffs (Python for ML vs. other
    considerations)

    -   Python

-   API design patterns (REST, GraphQL, or hybrid)

    -   REST

-   Stateful vs. stateless architecture considerations

    -   Stateless?

-   Authentication and authorization strategy

    -   Use Oauth

### **Frontend Approach**

-   SPA frameworks comparison (React, Vue, Angular)

    -   Streamlit

-   Data visualization libraries (D3.js, Chart.js, Plotly)

    -   Stremlit

### **Data Pipeline Architecture**

-   Batch vs. stream processing for different data types

-   Data ingestion workflow design

-   ETL/ELT approach for different data sources

-   Caching strategy at different system layers

-   Real-time processing requirements

### **Database Selection**

-   Time-series specialized DBs (InfluxDB, TimescaleDB) vs.
    general-purpose

-   Relational vs. NoSQL options for different data components

-   Sharding and partitioning strategies for historical data

-   Cold/hot storage approach for cost optimization

-   Backup and disaster recovery requirements

### **Infrastructure Considerations**

-   Cloud provider selection criteria (AWS, GCP, Azure)

-   Containerization and orchestration approach

-   Serverless vs. traditional deployment models

-   Auto-scaling policies and implementation

-   Monitoring and observability setup

## **3. Data Engineering Challenges (20 min)**

### **Historical data Api Ninja** 

### **https://www.api-ninjas.com/api/commodityprice**

## **4. Forecasting Algorithm Exploration (20 min)**

### **Time-series Forecasting Models**

-   Statistical models (ARIMA, SARIMA, ETS)

-   Machine learning approaches (Random Forest, XGBoost)

-   Deep learning options (LSTM, Transformer-based)

-   Ensemble methods for improved accuracy

-   Hybrid model architecture possibilities

### **Evaluation Metrics**

-   RMSE, MAE, MAPE comparison for different commodities

-   Accuracy vs. interpretability tradeoffs

-   Confidence interval calculation approaches

-   Custom metrics for specific use cases

-   Backtesting framework design

### **Computational Requirements**

-   Model training infrastructure needs

-   Inference latency considerations

-   GPU vs. CPU optimization

-   Batch prediction scheduling

-   Model versioning and lifecycle management

### **Implementation Strategy**

-   Build vs. buy decisions for algorithm components

-   Integration with ML frameworks (TensorFlow, PyTorch, scikit-learn)

-   Feature engineering pipeline design

-   Model explainability requirements

-   A/B testing infrastructure for model improvements

## **5. Development Workflow & Tools (15 min)**

### **Git Workflow and Branching**

-   Branch naming conventions

-   PR review requirements

-   Merge vs. rebase strategies

-   Release branch management

-   Hotfix procedures

### **CI/CD Pipeline Setup**

-   Automated testing requirements

-   Deployment environments (dev, staging, prod)

-   Infrastructure-as-code approach

-   Blue/green deployment strategy

-   Rollback procedures

### **Code Review Process**

-   Static analysis tools integration

-   Code quality metrics

-   Documentation requirements

-   Performance review standards

-   Security scanning integration

### **Testing Approach**

-   Unit testing framework selection

-   Integration testing strategy

-   Performance and load testing approach

-   Data validation testing

-   Automated UI testing requirements

## **6. Action Items & Division of Tasks (20 min)**

### **Prototype Assignments**

-   Backend core components ownership

-   Frontend framework selection and setup

-   Data pipeline initial design

-   ML model research responsibilities

-   Infrastructure configuration tasks

### **Research Tasks**

-   Technology stack evaluation criteria

-   Data source investigation assignments

-   Algorithm feasibility research

-   Performance benchmark requirements

-   Security and compliance investigation

### **Technical Documentation**

-   Architecture diagram responsibilities

-   API documentation standards

-   Development environment setup guides

-   Data dictionary and schema documentation

-   Knowledge base structure

### **Timeline and Milestones**

-   Proof-of-concept completion target

-   MVP feature set definition

-   Sprint planning approach

-   Initial release timeline

-   Review and feedback cycles
