# Commodity Price Forecasting Project - Presentation Transcript

## 1. Introduction

**[Team Member 1]:** Good morning everyone. I'm [Name], responsible for the frontend development and visualization aspects of our project.

**[Team Member 2]:** Hello, I'm [Name], and I've been working on the backend infrastructure and data collection for our system.

**[Team Member 3]:** And I'm [Name], focusing on the forecasting models and predictive analytics components.

**[Team Member 1]:** Today, we're excited to present our Commodity Price Forecasting system - a comprehensive toolset for fetching, analyzing, visualizing, and forecasting commodity price data with specific applications for agricultural policy making.

## 2. Why Commodity Price Forecasting Matters

**[Team Member 1]:** Let me start by explaining why visualization is critical in this domain. 

Commodities generate massive amounts of complex data that can be overwhelming for decision-makers. Our visualization system transforms this raw market data into actionable insights. The interactive dashboards we've built allow policymakers to identify price anomalies that may require intervention, visualize seasonal patterns critical for planning agricultural policies, and compare correlations between commodities to understand market dynamics.

For example, using our correlation heatmap, a policymaker could immediately see how coffee price movements relate to sugar prices, informing decisions about crop diversification programs. Our user-friendly interface ensures accessibility for both technical analysts and non-technical stakeholders, bridging the gap between data and decision-making.

**[Team Member 2]:** Building on that, reliable data collection forms the foundation for effective policy.

Our system provides a comprehensive historical database going back to 2000, giving essential context for evaluating current price movements. We've implemented automated data quality controls that ensure integrity for decision-making, flagging anomalies and preventing corrupted data from influencing critical decisions.

Rather than building complex data collection networks, we use efficient ETF-based proxies that accurately track true market movements. This approach gives us reliable data without the expense and complexity of direct market access. Importantly, our scalable architecture allows future integration with local market data, making it adaptable to different regions and markets.

**[Team Member 3]:** And ultimately, accurate forecasts directly impact agricultural planning and farmer protection.

Our multiple forecasting models capture different aspects of price dynamics - from seasonal patterns to complex non-linear relationships. We quantify uncertainty through confidence intervals, enabling proper risk assessment for policy interventions. The seasonal decomposition capabilities help identify optimal planting and harvest timing for farmers, potentially increasing yields and reducing waste.

Perhaps most importantly, our system provides early warning of price volatility, enabling proactive intervention measures before crises emerge. For agricultural commodities like corn or coffee, this can mean the difference between sustainable farmer incomes and economic hardship.

## 3. Technical Overview

### Frontend & Visualization

**[Team Member 1]:** Now let's dive deeper into the technical implementation, starting with our frontend and visualization components.

We've built our application using Streamlit, which provides an accessible, interactive interface without requiring complex web development. The interface features Yahoo Finance-style interactive charts with time range selectors ranging from 1 day to 5 years, allowing users to zoom in on relevant timeframes.

Our price comparison tools include normalization features for fair comparison between commodities with different price scales. For example, comparing gold at $2,000 per ounce with coffee at $2 per pound would be meaningless without normalization.

The correlation heatmaps reveal inter-commodity relationships at a glance, helping users identify which commodities tend to move together and which move independently.

We've organized specialized views for different commodity groups:
- Energy commodities like oil and natural gas
- Metals including gold, silver, and copper
- Agricultural products such as corn, wheat, soybeans, coffee, and sugar

These groupings allow for more focused analysis within related sectors. We've also implemented custom theming options, making the dashboard suitable for different presentation contexts, from internal analysis to stakeholder presentations.

All dashboard components update in real-time as users make selections, creating a responsive environment for exploration and analysis.

### Backend & Data Infrastructure

**[Team Member 2]:** Moving to our backend architecture, we've designed a robust system for data collection and management.

Our data collection architecture integrates with Yahoo Finance API through the yfinance library, providing reliable access to market data. We use ETF and ETN proxies as our data sources - for example, we track gold prices through GLD and oil through USO. This approach gives us access to liquid, representative markets without requiring direct commodity market access.

We've implemented automated data validation and error handling to ensure data quality, flagging outliers and handling missing datapoints appropriately.

For storage, we designed a SQLite database with an efficient schema optimized for time-series data. The structure includes unique constraints to prevent duplicate entries, ensuring data integrity. We chose SQLite for its portability - the entire database is a single file that can be easily deployed across different environments.

Our data processing pipeline includes incremental updates to minimize API usage, automatically fetching only the newest data since the last update. We generate statistical summaries for each commodity and handle missing values through appropriate interpolation techniques.

Key system features include one-click data refresh functionality, comprehensive error logging for troubleshooting, and optimized query performance for time-series analysis, ensuring that even with years of historical data, the application remains responsive.

### Forecasting Models

**[Team Member 3]:** The heart of our system lies in its forecasting capabilities, where we implement three complementary approaches.

First, we use Facebook Prophet, which is specialized for time series with strong seasonality. This is particularly valuable for agricultural commodities, which often follow yearly cycles due to growing seasons. Prophet automatically decomposes price series into trend and seasonal components, helping users understand the underlying patterns. It handles missing data and outliers robustly, maintaining accuracy even with imperfect datasets.

Our second approach is ARIMA - Autoregressive Integrated Moving Average - a classical statistical method for time series forecasting. ARIMA is particularly effective for short-term predictions where recent price movements strongly influence future prices. It identifies temporal dependencies in price data and provides statistical rigor with well-defined confidence intervals.

Finally, we implement LSTM (Long Short-Term Memory) neural networks, a deep learning approach that captures complex patterns that simpler models might miss. We've implemented these using TensorFlow for optimal performance and included an early stopping mechanism to prevent overfitting. This approach is especially powerful for commodities with complex price dynamics influenced by multiple factors.

All three models offer customizable forecast horizons from 30 to 365 days, transparent uncertainty quantification through confidence intervals, and model-specific visualizations of key components like trend and seasonality. This multi-model approach allows users to compare different forecasting techniques and select the one most appropriate for their specific commodity and timeframe.

## 4. Integrated Demonstration

**[Team Member 1]:** Let's walk through a complete workflow to demonstrate how these components work together.

We'll start by selecting a commodity and timeframe. From our dropdown menu, we can choose from 10 major commodities across energy, metals, and agriculture. For this demonstration, let's select Corn, which is relevant to many agricultural policy decisions.

We'll set the historical time range to the past 5 years to give us sufficient context. Immediately, we can examine price trends on our interactive chart, seeing how corn prices have behaved over this period.

**[Team Member 2]:** Now we can analyze this historical data in more depth. The system automatically calculates key statistics, showing the current price of corn, the year-to-date percentage change, and the 52-week price range.

Using our correlation tool, we can explore how corn prices relate to other agricultural commodities like wheat and soybeans. This reveals that corn and wheat have a correlation coefficient of 0.73, indicating strong price relationship - important information for policy planning across grain markets.

The system also identifies seasonal patterns automatically, showing that corn prices typically rise during planting season in spring and fall during harvest in autumn - a pattern critical for timing market interventions.

**[Team Member 3]:** With this understanding of historical patterns, we can now generate a price forecast.

For corn, which has strong seasonal patterns, we'll select the Prophet model which excels at capturing seasonality. We'll set a forecast horizon of 180 days, giving us a six-month outlook.

The system generates a forecast with confidence intervals, showing the expected price path and the range of potential outcomes. For corn, we can see the model predicts a 12% price increase over the next six months, with wider confidence intervals in the later months indicating growing uncertainty over time.

The component analysis reveals that this expected increase combines a positive trend component with a seasonal effect typical for the upcoming months.

**[Team Member 1]:** Now let's apply these insights to decision-making.

For a policy maker focused on food security, this forecast suggests potential upward price pressure that might affect consumers. The confidence intervals indicate a 10% chance that prices could rise by more than 20%, suggesting that contingency plans for price stabilization might be warranted.

The component analysis shows that much of the expected increase is seasonal, suggesting that this pattern might repeat in future years - valuable information for long-term planning.

This example demonstrates how our system moves from raw data to actionable insights through an integrated workflow of selection, analysis, forecasting, and interpretation.

## 5. Application for Ivorian Context

**[Team Member 1]:** Let's discuss how this system could be adapted specifically for the Ivorian context.

For visualization and decision support, we envision customized dashboards focusing on Ivorian cocoa, coffee, and cashew markets - the country's key agricultural exports. The interface could be extended with multilingual support in French and regional languages to ensure accessibility for diverse stakeholders.

We recognize that mobile access is crucial in many regions, so we've designed the visualizations to be mobile-optimized for field use by agricultural extension workers. We could also develop simplified visualization modes specifically for community meetings and farmer engagement, making the data accessible even to those with limited technical background.

**[Team Member 2]:** On the data side, our framework is ready for integration with local market price data from regional collection points across Côte d'Ivoire.

We could establish connections to existing Ivorian agricultural databases, ensuring our system complements rather than duplicates current efforts. The import/export functionality we've built ensures compatibility with government systems and data formats.

We've also considered strategies for handling data gaps in remote agricultural regions, including statistical methods for imputation and uncertainty quantification when data is sparse.

**[Team Member 3]:** For the forecasting models, we would need specific calibration for Ivorian commodities and conditions.

This would include adaptation of seasonal components to West African climate patterns, which differ significantly from those in global commodity markets. We would incorporate Harmattan weather effects on agricultural production, as this seasonal dry wind has substantial impact on growing conditions.

Our models would be tuned to account for regional factors affecting price seasonality, such as the specific timing of rainy seasons and harvest periods in Côte d'Ivoire. We would also analyze cross-commodity effects specific to the Ivorian agricultural basket, particularly the relationships between cocoa, coffee, and other crops frequently grown together.

## 6. Challenges & Future Directions

**[Team Member 1]:** Looking ahead, we see several opportunities for enhancing our system, starting with UX improvements.

A dedicated mobile application would enable field usage without requiring internet browsers. We'd implement offline functionality for areas with limited connectivity, allowing data collection and basic analysis even without constant internet access.

Multi-language support is essential for diverse stakeholder groups across different regions, as is the development of simplified visualization modes specifically designed for farmer education and engagement.

We would also implement community feedback mechanisms to drive continuous improvement, ensuring the system evolves based on real user needs.

**[Team Member 2]:** For data expansion, we envision integration with local market price collection networks to supplement our ETF-based data with ground-level information.

A real-time price alert system could enable rapid intervention when markets show unusual volatility. Incorporating weather data would enhance supply forecasting, especially for weather-sensitive crops.

Connections to global commodity exchanges would enable price arbitrage analysis, identifying opportunities for beneficial trade. We also plan historical data extension beyond ETF timeframes, possibly using alternative data sources to extend our analysis further back in time.

**[Team Member 3]:** Finally, we see significant potential for enhanced modeling approaches.

Commodity-specific model tuning would improve accuracy for key Ivorian products. Integrating supply-side factors like production forecasts and inventory levels would create more comprehensive models that capture both demand and supply dynamics.

External variable incorporation - particularly weather patterns and global economic indicators - would improve forecast accuracy by capturing more causal factors. We could implement ensemble methods that combine the strengths of multiple models, potentially outperforming any single approach.

And finally, we could develop specialized models for different time horizons, recognizing that the factors driving short-term price movements often differ from those driving long-term trends.

## 7. Conclusion

**[Team Member 1]:** In summary, our system provides a complete solution for commodity price analysis and forecasting.

**[Team Member 2]:** We've built powerful tools that can help policymakers stabilize markets and protect farmer incomes through better information and foresight.

**[Team Member 3]:** The flexible architecture allows customization for specific Ivorian needs, making it adaptable to local conditions and priorities.

**[Team Member 1]:** We see potential for significant impact on agricultural planning and food security through better price prediction and market understanding.

**[Team Member 2]:** Our proposed next step is a pilot implementation focusing on key Ivorian commodities, particularly cocoa and coffee, to demonstrate the system's value in a real-world context.

**[Team Member 3]:** Thank you for your attention. We welcome any questions about our approach and how it might be applied to your specific needs.