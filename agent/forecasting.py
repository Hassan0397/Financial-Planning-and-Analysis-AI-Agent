"""
Forecasting Module
Time series forecasting for FP&A metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import forecasting libraries (some might be optional)
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    st.warning("⚠️ Prophet not installed. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    sklearn_available = True
except ImportError:
    sklearn_available = False


class FinancialForecaster:
    """
    Time series forecasting for financial metrics
    """
    
    def __init__(self):
        self.forecast_models = {}
        self.forecast_results = {}
        self.model_performance = {}
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> Dict[str, bool]:
        """Check which forecasting models are available"""
        models = {
            'Prophet': prophet_available,
            'ARIMA': statsmodels_available,
            'Linear Regression': sklearn_available,
            'Random Forest': sklearn_available,
            'Exponential Smoothing': True,  # Simple implementation
            'Moving Average': True,  # Simple implementation
            'Seasonal Naive': True  # Simple implementation
        }
        return models
    
    def display_forecasting_interface(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Main Streamlit interface for forecasting
        """
        st.subheader("🔮 Financial Forecasting")
        
        if not dataframes:
            st.warning("⚠️ No data available. Please upload and analyze data first.")
            return {}
        
        # Get current data
        current_data = dataframes
        
        # File selection
        primary_file = st.selectbox(
            "Select file for forecasting:",
            options=list(current_data.keys()),
            help="This file should contain time series data"
        )
        
        if not primary_file:
            return {}
        
        df = current_data[primary_file]
        
        # Check for date column
        date_cols = self._detect_date_columns(df)
        if not date_cols:
            st.error("❌ No date column found. Forecasting requires time series data.")
            st.info("**Tip:** Ensure your data has a column with dates (e.g., 'Date', 'Month', 'Year')")
            return {}
        
        date_col = date_cols[0]
        
        # Ensure date column is datetime
        df_forecast = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_forecast[date_col]):
            df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors='coerce')
        
        # Sort by date
        df_forecast = df_forecast.sort_values(date_col)
        
        # Detect numeric columns for forecasting
        numeric_cols = df_forecast.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("❌ No numeric columns found for forecasting.")
            return {}
        
        # Create tabs for different forecasting sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Single Series Forecast", 
            "📊 Multi-Series Forecast", 
            "🔍 Model Comparison",
            "📋 Forecast Scenarios",
            "⚙️ Advanced Settings"
        ])
        
        forecast_results = {}
        
        with tab1:
            forecast_results['single_series'] = self._single_series_forecast(df_forecast, date_col, numeric_cols)
        
        with tab2:
            forecast_results['multi_series'] = self._multi_series_forecast(df_forecast, date_col, numeric_cols)
        
        with tab3:
            forecast_results['model_comparison'] = self._model_comparison(df_forecast, date_col, numeric_cols)
        
        with tab4:
            forecast_results['scenarios'] = self._forecast_scenarios(df_forecast, date_col, numeric_cols)
        
        with tab5:
            forecast_results['advanced'] = self._advanced_settings(df_forecast, date_col, numeric_cols)
        
        # Store results
        self.forecast_results = forecast_results
        
        return forecast_results
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect date columns in dataframe"""
        date_cols = []
        
        for col in df.columns:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            else:
                # Check by name
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'quarter', 'period']):
                    date_cols.append(col)
        
        return date_cols
    
    def _single_series_forecast(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """
        Forecast a single time series
        """
        st.markdown("### 📈 Single Series Forecast")
        
        results = {}
        
        # Metric selection
        metric = st.selectbox(
            "Select metric to forecast:",
            options=numeric_cols,
            key='single_metric'
        )
        
        # Prepare time series
        ts_data = self._prepare_time_series(df, date_col, metric)
        
        if ts_data.empty:
            st.warning("Not enough data for forecasting.")
            return results
        
        # Display time series overview
        self._display_series_overview(ts_data, metric)
        
        # Forecast horizon
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_periods = st.number_input(
                "Forecast periods:",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of future periods to forecast"
            )
        
        with col2:
            freq = st.selectbox(
                "Data frequency:",
                options=['D (Daily)', 'W (Weekly)', 'M (Monthly)', 'Q (Quarterly)', 'Y (Yearly)'],
                index=0
            )
            freq_map = {'D (Daily)': 'D', 'W (Weekly)': 'W', 'M (Monthly)': 'M', 'Q (Quarterly)': 'Q', 'Y (Yearly)': 'Y'}
            freq_code = freq_map[freq]
        
        # Model selection
        available_model_names = [name for name, avail in self.available_models.items() if avail]
        if not available_model_names:
            st.error("No forecasting models available. Please install required packages.")
            return results
        
        model_name = st.selectbox(
            "Select forecasting model:",
            options=available_model_names,
            help="Choose the forecasting algorithm to use"
        )
        
        # Additional model parameters
        if model_name == 'Prophet':
            growth = st.selectbox("Growth trend:", ['linear', 'logistic'])
            seasonality = st.checkbox("Include seasonality", value=True)
            holidays = st.checkbox("Include holidays", value=False)
        
        elif model_name == 'ARIMA':
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("AR order (p):", 0, 5, 1)
            with col2:
                d = st.slider("Difference order (d):", 0, 2, 1)
            with col3:
                q = st.slider("MA order (q):", 0, 5, 1)
        
        elif model_name in ['Linear Regression', 'Random Forest']:
            include_lags = st.checkbox("Include lag features", value=True)
            if include_lags:
                n_lags = st.slider("Number of lag periods:", 1, 12, 3)
        
        # Confidence interval
        confidence_level = st.slider(
            "Confidence interval:",
            min_value=50,
            max_value=99,
            value=95,
            step=5,
            help="Confidence level for prediction intervals"
        )
        
        # Run forecast button
        if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_name} model..."):
                # Run forecast
                forecast_result = self._run_forecast(
                    ts_data=ts_data,
                    model_name=model_name,
                    forecast_periods=forecast_periods,
                    freq=freq_code,
                    confidence_level=confidence_level,
                    model_params={
                        'growth': growth if model_name == 'Prophet' else None,
                        'p': p if model_name == 'ARIMA' else None,
                        'd': d if model_name == 'ARIMA' else None,
                        'q': q if model_name == 'ARIMA' else None,
                        'include_lags': include_lags if model_name in ['Linear Regression', 'Random Forest'] else None,
                        'n_lags': n_lags if model_name in ['Linear Regression', 'Random Forest'] else None
                    }
                )
                
                if forecast_result:
                    results = forecast_result
                    
                    # Display forecast results
                    self._display_forecast_results(forecast_result, metric, model_name)
                    
                    # Store model
                    model_key = f"{metric}_{model_name}"
                    self.forecast_models[model_key] = forecast_result.get('model', None)
                    self.model_performance[model_key] = forecast_result.get('performance', {})
        
        return results
    
    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, metric: str) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        # Ensure date is datetime
        ts_df = df[[date_col, metric]].copy()
        ts_df = ts_df.dropna()
        
        if ts_df.empty:
            return pd.DataFrame()
        
        # Aggregate by date (sum)
        ts_agg = ts_df.groupby(date_col)[metric].sum().reset_index()
        
        # Sort by date
        ts_agg = ts_agg.sort_values(date_col)
        
        # Ensure regular frequency (fill missing dates)
        if pd.api.types.is_datetime64_any_dtype(ts_agg[date_col]):
            full_date_range = pd.date_range(
                start=ts_agg[date_col].min(),
                end=ts_agg[date_col].max(),
                freq='D'
            )
            
            ts_complete = pd.DataFrame({date_col: full_date_range})
            ts_complete = ts_complete.merge(ts_agg, on=date_col, how='left')
            
            # Forward fill for missing values
            ts_complete[metric] = ts_complete[metric].fillna(method='ffill')
            
            return ts_complete
        
        return ts_agg
    
    def _display_series_overview(self, ts_data: pd.DataFrame, metric: str):
        """Display time series overview"""
        if ts_data.empty:
            return
        
        date_col = ts_data.columns[0]
        
        # Calculate statistics
        stats = {
            'Start Date': ts_data[date_col].min().strftime('%Y-%m-%d') if hasattr(ts_data[date_col].min(), 'strftime') else str(ts_data[date_col].min()),
            'End Date': ts_data[date_col].max().strftime('%Y-%m-%d') if hasattr(ts_data[date_col].max(), 'strftime') else str(ts_data[date_col].max()),
            'Data Points': len(ts_data),
            f'Total {metric}': f"${ts_data[metric].sum():,.0f}",
            f'Average {metric}': f"${ts_data[metric].mean():,.0f}",
            f'Std Dev {metric}': f"${ts_data[metric].std():,.0f}",
            'Trend': self._calculate_trend(ts_data[metric])
        }
        
        # Display stats
        cols = st.columns(4)
        for i, (key, value) in enumerate(stats.items()):
            with cols[i % 4]:
                st.metric(key, value)
        
        # Time series plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ts_data[date_col],
            y=ts_data[metric],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        window = min(30, len(ts_data) // 4)
        if window > 1:
            ts_data['MA'] = ts_data[metric].rolling(window=window, center=True).mean()
            fig.add_trace(go.Scatter(
                x=ts_data[date_col],
                y=ts_data['MA'],
                mode='lines',
                name=f'{window}-period MA',
                line=dict(color='orange', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{metric} Time Series",
            xaxis_title="Date",
            yaxis_title=metric,
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonality decomposition
        if len(ts_data) >= 2 * 30:  # At least 2 months of daily data
            if st.checkbox("Show seasonality decomposition"):
                self._decompose_time_series(ts_data, metric)
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return "Insufficient data"
        
        # Simple linear trend
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return "Insufficient data"
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0:
            return f"↗ Up ({slope:.2f}/period)"
        elif slope < 0:
            return f"↘ Down ({slope:.2f}/period)"
        else:
            return "→ Flat"
    
    def _decompose_time_series(self, ts_data: pd.DataFrame, metric: str):
        """Decompose time series into trend, seasonality, and residual"""
        date_col = ts_data.columns[0]
        
        # Ensure regular frequency
        ts_series = ts_data.set_index(date_col)[metric]
        
        # Determine period based on data
        if len(ts_series) >= 365:  # Daily data with year
            period = 365
        elif len(ts_series) >= 52:  # Weekly data
            period = 52
        elif len(ts_series) >= 12:  # Monthly data
            period = 12
        elif len(ts_series) >= 4:  # Quarterly data
            period = 4
        else:
            st.info("Not enough data for decomposition")
            return
        
        try:
            decomposition = seasonal_decompose(
                ts_series.dropna(),
                model='additive',
                period=min(period, len(ts_series) // 2)
            )
            
            # Plot decomposition
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.05
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'),
                row=4, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(matches='x')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stationarity test
            if statsmodels_available:
                st.markdown("##### 📊 Stationarity Test (ADF)")
                adf_result = adfuller(ts_series.dropna())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                with col2:
                    st.metric("p-value", f"{adf_result[1]:.4f}")
                with col3:
                    is_stationary = adf_result[1] < 0.05
                    st.metric(
                        "Stationary", 
                        "✅ Yes" if is_stationary else "❌ No",
                        delta_color="normal" if is_stationary else "inverse"
                    )
                
                if not is_stationary:
                    st.info("Series is not stationary. Consider differencing for better forecasts.")
        
        except Exception as e:
            st.warning(f"Decomposition failed: {str(e)}")
    
    def _run_forecast(self, ts_data: pd.DataFrame, model_name: str, forecast_periods: int, 
                     freq: str, confidence_level: int, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run forecasting with specified model"""
        date_col = ts_data.columns[0]
        metric = ts_data.columns[1]
        
        results = {
            'model_name': model_name,
            'forecast_periods': forecast_periods,
            'confidence_level': confidence_level,
            'actual_data': ts_data.copy()
        }
        
        try:
            if model_name == 'Prophet' and prophet_available:
                forecast_result = self._prophet_forecast(ts_data, date_col, metric, forecast_periods, model_params)
            
            elif model_name == 'ARIMA' and statsmodels_available:
                forecast_result = self._arima_forecast(ts_data, date_col, metric, forecast_periods, model_params)
            
            elif model_name == 'Linear Regression' and sklearn_available:
                forecast_result = self._regression_forecast(ts_data, date_col, metric, forecast_periods, model_params, model_type='linear')
            
            elif model_name == 'Random Forest' and sklearn_available:
                forecast_result = self._regression_forecast(ts_data, date_col, metric, forecast_periods, model_params, model_type='rf')
            
            elif model_name == 'Exponential Smoothing':
                forecast_result = self._exponential_smoothing_forecast(ts_data, date_col, metric, forecast_periods)
            
            elif model_name == 'Moving Average':
                forecast_result = self._moving_average_forecast(ts_data, date_col, metric, forecast_periods)
            
            elif model_name == 'Seasonal Naive':
                forecast_result = self._seasonal_naive_forecast(ts_data, date_col, metric, forecast_periods)
            
            else:
                st.error(f"Model '{model_name}' not available or not implemented.")
                return {}
            
            results.update(forecast_result)
            
            # Calculate performance metrics if we have test data
            if 'test_data' in forecast_result and 'predictions' in forecast_result:
                performance = self._calculate_performance_metrics(
                    forecast_result['test_data'],
                    forecast_result['predictions']
                )
                results['performance'] = performance
                self._display_performance_metrics(performance)
            
            return results
        
        except Exception as e:
            st.error(f"Forecast failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return {}
    
    def _prophet_forecast(self, ts_data: pd.DataFrame, date_col: str, metric: str, 
                         forecast_periods: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast using Facebook Prophet"""
        # Prepare data for Prophet
        prophet_df = ts_data[[date_col, metric]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Create and fit model
        model = Prophet(
            growth=params.get('growth', 'linear'),
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=params.get('confidence_level', 0.95) / 100
        )
        
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods, freq='D')
        
        # Forecast
        forecast = model.predict(future)
        
        # Prepare results
        forecast_dates = forecast['ds'].iloc[-forecast_periods:]
        forecast_values = forecast['yhat'].iloc[-forecast_periods:]
        forecast_lower = forecast['yhat_lower'].iloc[-forecast_periods:]
        forecast_upper = forecast['yhat_upper'].iloc[-forecast_periods:]
        
        results = {
            'model': model,
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values.values,
            'forecast_lower': forecast_lower.values,
            'forecast_upper': forecast_upper.values,
            'forecast_df': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
        }
        
        return results
    
    def _arima_forecast(self, ts_data: pd.DataFrame, date_col: str, metric: str, 
                       forecast_periods: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast using ARIMA"""
        from statsmodels.tsa.arima.model import ARIMA
        
        # Prepare series
        series = ts_data.set_index(date_col)[metric].dropna()
        
        # Split for testing
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Fit ARIMA model
        model = ARIMA(train, order=(params['p'], params['d'], params['q']))
        model_fit = model.fit()
        
        # Forecast
        forecast_result = model_fit.forecast(steps=len(test) + forecast_periods)
        forecast_values = forecast_result[-forecast_periods:]
        
        # Create future dates
        last_date = ts_data[date_col].max()
        if pd.api.types.is_datetime64_any_dtype(last_date):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        else:
            future_dates = list(range(1, forecast_periods + 1))
        
        results = {
            'model': model_fit,
            'test_data': test,
            'predictions': forecast_result[:len(test)],
            'forecast_dates': future_dates,
            'forecast_values': forecast_values.values,
            'forecast_df': pd.DataFrame({
                'date': future_dates,
                'forecast': forecast_values.values
            })
        }
        
        return results
    
    def _regression_forecast(self, ts_data: pd.DataFrame, date_col: str, metric: str, 
                           forecast_periods: int, params: Dict[str, Any], model_type: str = 'linear') -> Dict[str, Any]:
        """Forecast using regression models"""
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features
        series = ts_data[metric].values
        
        if params.get('include_lags', False):
            n_lags = params.get('n_lags', 3)
            
            # Create lag features
            X, y = [], []
            for i in range(n_lags, len(series)):
                X.append(series[i-n_lags:i])
                y.append(series[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train model
            if model_type == 'linear':
                model = LinearRegression()
            else:  # random forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Predict on test
            y_pred = model.predict(X_test)
            
            # Forecast future
            last_window = series[-n_lags:]
            future_predictions = []
            
            for _ in range(forecast_periods):
                next_pred = model.predict(last_window.reshape(1, -1))[0]
                future_predictions.append(next_pred)
                last_window = np.append(last_window[1:], next_pred)
            
        else:
            # Simple time trend
            X = np.arange(len(series)).reshape(-1, 1)
            y = series
            
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            if model_type == 'linear':
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Forecast future
            future_indices = np.arange(len(series), len(series) + forecast_periods).reshape(-1, 1)
            future_predictions = model.predict(future_indices).flatten()
        
        # Create future dates
        last_date = ts_data[date_col].max()
        if pd.api.types.is_datetime64_any_dtype(last_date):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        else:
            future_dates = list(range(1, forecast_periods + 1))
        
        results = {
            'model': model,
            'test_data': y_test,
            'predictions': y_pred,
            'forecast_dates': future_dates,
            'forecast_values': future_predictions,
            'forecast_df': pd.DataFrame({
                'date': future_dates,
                'forecast': future_predictions
            })
        }
        
        return results
    
    def _exponential_smoothing_forecast(self, ts_data: pd.DataFrame, date_col: str, 
                                      metric: str, forecast_periods: int) -> Dict[str, Any]:
        """Simple exponential smoothing forecast"""
        series = ts_data[metric].values
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = [series[0]]
        
        for i in range(1, len(series)):
            smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast (flat line at last smoothed value)
        last_value = smoothed[-1]
        future_predictions = [last_value] * forecast_periods
        
        # Create future dates
        last_date = ts_data[date_col].max()
        if pd.api.types.is_datetime64_any_dtype(last_date):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        else:
            future_dates = list(range(1, forecast_periods + 1))
        
        return {
            'forecast_dates': future_dates,
            'forecast_values': future_predictions,
            'forecast_df': pd.DataFrame({
                'date': future_dates,
                'forecast': future_predictions
            })
        }
    
    def _moving_average_forecast(self, ts_data: pd.DataFrame, date_col: str, 
                               metric: str, forecast_periods: int) -> Dict[str, Any]:
        """Moving average forecast"""
        series = ts_data[metric].values
        
        # Use last N periods average
        window = min(7, len(series) // 4)
        last_avg = np.mean(series[-window:])
        
        future_predictions = [last_avg] * forecast_periods
        
        # Create future dates
        last_date = ts_data[date_col].max()
        if pd.api.types.is_datetime64_any_dtype(last_date):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        else:
            future_dates = list(range(1, forecast_periods + 1))
        
        return {
            'forecast_dates': future_dates,
            'forecast_values': future_predictions,
            'forecast_df': pd.DataFrame({
                'date': future_dates,
                'forecast': future_predictions
            })
        }
    
    def _seasonal_naive_forecast(self, ts_data: pd.DataFrame, date_col: str, 
                               metric: str, forecast_periods: int) -> Dict[str, Any]:
        """Seasonal naive forecast"""
        series = ts_data[metric].values
        
        # Assume yearly seasonality (365 days)
        season_length = min(365, len(series))
        
        future_predictions = []
        for i in range(forecast_periods):
            idx = (len(series) - season_length + i) % season_length
            future_predictions.append(series[idx])
        
        # Create future dates
        last_date = ts_data[date_col].max()
        if pd.api.types.is_datetime64_any_dtype(last_date):
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        else:
            future_dates = list(range(1, forecast_periods + 1))
        
        return {
            'forecast_dates': future_dates,
            'forecast_values': future_predictions,
            'forecast_df': pd.DataFrame({
                'date': future_dates,
                'forecast': future_predictions
            })
        }
    
    def _calculate_performance_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast performance metrics"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return {}
        
        # Calculate errors
        errors = actual - predicted
        abs_errors = np.abs(errors)
        
        # Metrics
        metrics = {
            'MAE': np.mean(abs_errors),  # Mean Absolute Error
            'MSE': np.mean(errors ** 2),  # Mean Squared Error
            'RMSE': np.sqrt(np.mean(errors ** 2)),  # Root Mean Squared Error
            'MAPE': np.mean(np.abs(errors / actual)) * 100 if np.all(actual != 0) else np.nan,  # Mean Absolute Percentage Error
            'R2': 1 - np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2) if np.var(actual) > 0 else np.nan
        }
        
        return metrics
    
    def _display_performance_metrics(self, metrics: Dict[str, float]):
        """Display forecast performance metrics"""
        st.markdown("##### 📊 Forecast Performance")
        
        if not metrics:
            st.info("No performance metrics available")
            return
        
        cols = st.columns(len(metrics))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i]:
                if pd.isna(metric_value):
                    st.metric(metric_name, "N/A")
                elif metric_name in ['MAE', 'MSE', 'RMSE']:
                    st.metric(metric_name, f"{metric_value:,.2f}")
                elif metric_name == 'MAPE':
                    st.metric(metric_name, f"{metric_value:.1f}%")
                elif metric_name == 'R2':
                    st.metric(metric_name, f"{metric_value:.3f}")
                else:
                    st.metric(metric_name, f"{metric_value:.4f}")
    
    def _display_forecast_results(self, forecast_result: Dict[str, Any], metric: str, model_name: str):
        """Display forecast results visually"""
        # Combine actual and forecast data
        actual_dates = forecast_result['actual_data'].iloc[:, 0]
        actual_values = forecast_result['actual_data'].iloc[:, 1]
        forecast_dates = forecast_result['forecast_dates']
        forecast_values = forecast_result['forecast_values']
        
        # Create visualization
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence interval if available
        if 'forecast_lower' in forecast_result and 'forecast_upper' in forecast_result:
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates)[::-1],
                y=list(forecast_result['forecast_upper']) + list(forecast_result['forecast_lower'])[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{forecast_result['confidence_level']}% Confidence"
            ))
        
        fig.update_layout(
            title=f"{metric} Forecast using {model_name}",
            xaxis_title="Date",
            yaxis_title=metric,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast values
        st.markdown("##### 📋 Forecast Values")
        
        if 'forecast_df' in forecast_result:
            forecast_df = forecast_result['forecast_df'].copy()
            
            # Format the forecast values
            if metric.lower() in ['revenue', 'cost', 'profit', 'price']:
                forecast_df['forecast'] = forecast_df['forecast'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(forecast_df, use_container_width=True)
        
        # Summary statistics
        st.markdown("##### 📊 Forecast Summary")
        
        if len(forecast_values) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_forecast = np.sum(forecast_values)
                st.metric(
                    "Total Forecast",
                    f"${total_forecast:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{total_forecast:,.0f}"
                )
            
            with col2:
                avg_forecast = np.mean(forecast_values)
                st.metric(
                    "Average Forecast",
                    f"${avg_forecast:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{avg_forecast:,.0f}"
                )
            
            with col3:
                growth = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100) if forecast_values[0] != 0 else 0
                st.metric(
                    "Forecast Growth",
                    f"{growth:+.1f}%"
                )
            
            with col4:
                volatility = np.std(forecast_values)
                st.metric(
                    "Forecast Volatility",
                    f"${volatility:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{volatility:,.0f}"
                )
    
    def _multi_series_forecast(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Forecast multiple time series"""
        st.markdown("### 📊 Multi-Series Forecast")
        
        results = {}
        
        st.info("Multi-series forecasting coming soon. Currently use Single Series Forecast for individual metrics.")
        
        return results
    
    def _model_comparison(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Compare different forecasting models"""
        st.markdown("### 🔍 Model Comparison")
        
        results = {}
        
        # Metric selection
        metric = st.selectbox(
            "Select metric for comparison:",
            options=numeric_cols,
            key='compare_metric'
        )
        
        # Prepare time series
        ts_data = self._prepare_time_series(df, date_col, metric)
        
        if ts_data.empty or len(ts_data) < 20:
            st.warning("Not enough data for model comparison.")
            return results
        
        # Available models
        available_models = [name for name, avail in self.available_models.items() if avail]
        
        if len(available_models) < 2:
            st.warning("Need at least 2 models for comparison.")
            return results
        
        # Model selection
        selected_models = st.multiselect(
            "Select models to compare:",
            options=available_models,
            default=available_models[:min(3, len(available_models))]
        )
        
        if len(selected_models) < 2:
            return results
        
        # Forecast horizon
        forecast_periods = st.slider(
            "Forecast periods for comparison:",
            min_value=5,
            max_value=100,
            value=30,
            key='compare_periods'
        )
        
        # Test set size
        test_size = st.slider(
            "Test set size (for backtesting):",
            min_value=5,
            max_value=min(100, len(ts_data) // 2),
            value=min(30, len(ts_data) // 4),
            help="Number of periods to use for testing model accuracy"
        )
        
        if st.button("🔬 Compare Models", type="primary", use_container_width=True):
            with st.spinner("Running model comparison..."):
                comparison_results = self._run_model_comparison(
                    ts_data=ts_data,
                    metric=metric,
                    models=selected_models,
                    forecast_periods=forecast_periods,
                    test_size=test_size
                )
                
                results = comparison_results
                
                # Display comparison results
                if comparison_results:
                    self._display_model_comparison(comparison_results, metric)
        
        return results
    
    def _run_model_comparison(self, ts_data: pd.DataFrame, metric: str, models: List[str], 
                            forecast_periods: int, test_size: int) -> Dict[str, Any]:
        """Run comparison between multiple models"""
        date_col = ts_data.columns[0]
        series = ts_data[metric].values
        
        # Split data
        train_size = len(series) - test_size
        train = series[:train_size]
        test = series[train_size:]
        
        results = {
            'models': models,
            'test_data': test,
            'model_results': {},
            'performance_metrics': {}
        }
        
        # Train and evaluate each model
        for model_name in models:
            try:
                # Simple model parameters
                params = {
                    'p': 1, 'd': 1, 'q': 1,  # ARIMA defaults
                    'growth': 'linear',  # Prophet default
                    'include_lags': True,
                    'n_lags': 3
                }
                
                # Create training dataframe
                train_dates = ts_data.iloc[:train_size][date_col]
                train_df = pd.DataFrame({date_col: train_dates, metric: train})
                
                # Run forecast
                forecast_result = self._run_forecast(
                    ts_data=train_df,
                    model_name=model_name,
                    forecast_periods=test_size + forecast_periods,
                    freq='D',
                    confidence_level=95,
                    model_params=params
                )
                
                if forecast_result:
                    # Get predictions for test period
                    if 'predictions' in forecast_result:
                        predictions = forecast_result['predictions'][:test_size]
                    elif 'forecast_values' in forecast_result:
                        predictions = forecast_result['forecast_values'][:test_size]
                    else:
                        predictions = []
                    
                    # Calculate performance
                    if len(predictions) == len(test):
                        performance = self._calculate_performance_metrics(test, predictions)
                        
                        results['model_results'][model_name] = forecast_result
                        results['performance_metrics'][model_name] = performance
                    
            except Exception as e:
                st.warning(f"Model {model_name} failed: {str(e)}")
        
        return results
    
    def _display_model_comparison(self, comparison_results: Dict[str, Any], metric: str):
        """Display model comparison results"""
        # Performance comparison table
        st.markdown("##### 📊 Model Performance Comparison")
        
        perf_metrics = comparison_results['performance_metrics']
        
        if not perf_metrics:
            st.info("No performance metrics available for comparison")
            return
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in perf_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight best performing model for each metric
        def highlight_best(s):
            if s.name in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                best_idx = s.idxmin()
            elif s.name == 'R2':
                best_idx = s.idxmax()
            else:
                return [''] * len(s)
            
            return ['background-color: lightgreen' if i == best_idx else '' for i in range(len(s))]
        
        styled_df = comparison_df.style.apply(highlight_best, subset=pd.IndexSlice[:, comparison_df.columns[1:]])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visual comparison
        st.markdown("##### 📈 Forecast Comparison")
        
        fig = go.Figure()
        
        # Add test data
        test_data = comparison_results['test_data']
        x_test = list(range(len(test_data)))
        
        fig.add_trace(go.Scatter(
            x=x_test,
            y=test_data,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=3)
        ))
        
        # Add model predictions
        colors = px.colors.qualitative.Set3
        
        for i, (model_name, model_result) in enumerate(comparison_results['model_results'].items()):
            if 'predictions' in model_result:
                predictions = model_result['predictions'][:len(test_data)]
                
                fig.add_trace(go.Scatter(
                    x=x_test,
                    y=predictions,
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f"Model Comparison for {metric}",
            xaxis_title="Period",
            yaxis_title=metric,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if len(perf_metrics) > 0:
            st.markdown("##### 🏆 Recommended Model")
            
            # Find best model based on RMSE
            best_model = min(perf_metrics.items(), key=lambda x: x[1].get('RMSE', float('inf')))[0]
            best_rmse = perf_metrics[best_model].get('RMSE', 0)
            
            st.success(f"**{best_model}** is recommended with RMSE of {best_rmse:,.2f}")
    
    def _forecast_scenarios(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Create forecast scenarios with different assumptions"""
        st.markdown("### 📋 Forecast Scenarios")
        
        results = {}
        
        # Metric selection
        metric = st.selectbox(
            "Select metric for scenarios:",
            options=numeric_cols,
            key='scenario_metric'
        )
        
        # Base forecast
        ts_data = self._prepare_time_series(df, date_col, metric)
        
        if ts_data.empty:
            return results
        
        # Create scenarios
        st.markdown("#### 🎯 Scenario Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_growth = st.slider(
                "Base Growth %",
                min_value=-50.0,
                max_value=200.0,
                value=5.0,
                step=0.5,
                help="Base monthly growth rate"
            )
        
        with col2:
            optimistic_adjustment = st.slider(
                "Optimistic Adjustment %",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                help="Additional growth for optimistic scenario"
            )
        
        with col3:
            pessimistic_adjustment = st.slider(
                "Pessimistic Adjustment %",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                help="Reduction for pessimistic scenario"
            )
        
        # Forecast horizon
        forecast_periods = st.slider(
            "Forecast periods:",
            min_value=1,
            max_value=36,
            value=12,
            key='scenario_periods'
        )
        
        if st.button("📊 Generate Scenarios", type="primary", use_container_width=True):
            # Generate scenarios
            last_value = ts_data[metric].iloc[-1]
            
            # Base scenario
            base_forecast = []
            for i in range(forecast_periods):
                growth_factor = (1 + base_growth / 100) ** (i + 1)
                base_forecast.append(last_value * growth_factor)
            
            # Optimistic scenario
            optimistic_forecast = []
            for i in range(forecast_periods):
                growth_factor = (1 + (base_growth + optimistic_adjustment) / 100) ** (i + 1)
                optimistic_forecast.append(last_value * growth_factor)
            
            # Pessimistic scenario
            pessimistic_forecast = []
            for i in range(forecast_periods):
                growth_factor = (1 + (base_growth - pessimistic_adjustment) / 100) ** (i + 1)
                pessimistic_forecast.append(last_value * growth_factor)
            
            # Create results
            last_date = ts_data[date_col].max()
            if pd.api.types.is_datetime64_any_dtype(last_date):
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='M')
            else:
                future_dates = list(range(1, forecast_periods + 1))
            
            results = {
                'scenario_dates': future_dates,
                'base_forecast': base_forecast,
                'optimistic_forecast': optimistic_forecast,
                'pessimistic_forecast': pessimistic_forecast,
                'scenario_df': pd.DataFrame({
                    'Date': future_dates,
                    'Base Scenario': base_forecast,
                    'Optimistic Scenario': optimistic_forecast,
                    'Pessimistic Scenario': pessimistic_forecast
                })
            }
            
            # Display scenarios
            self._display_scenario_results(results, metric)
        
        return results
    
    def _display_scenario_results(self, scenario_results: Dict[str, Any], metric: str):
        """Display scenario forecast results"""
        # Scenario comparison chart
        fig = go.Figure()
        
        # Add scenarios
        fig.add_trace(go.Scatter(
            x=scenario_results['scenario_dates'],
            y=scenario_results['base_forecast'],
            mode='lines',
            name='Base Scenario',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_results['scenario_dates'],
            y=scenario_results['optimistic_forecast'],
            mode='lines',
            name='Optimistic',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=scenario_results['scenario_dates'],
            y=scenario_results['pessimistic_forecast'],
            mode='lines',
            name='Pessimistic',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Fill between scenarios
        fig.add_trace(go.Scatter(
            x=list(scenario_results['scenario_dates']) + list(scenario_results['scenario_dates'])[::-1],
            y=list(scenario_results['optimistic_forecast']) + list(scenario_results['pessimistic_forecast'])[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Scenario Range'
        ))
        
        fig.update_layout(
            title=f"{metric} Forecast Scenarios",
            xaxis_title="Date",
            yaxis_title=metric,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario summary
        st.markdown("##### 📊 Scenario Summary")
        
        scenario_df = scenario_results['scenario_df'].copy()
        
        # Calculate totals
        base_total = sum(scenario_results['base_forecast'])
        optimistic_total = sum(scenario_results['optimistic_forecast'])
        pessimistic_total = sum(scenario_results['pessimistic_forecast'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Base Scenario Total",
                f"${base_total:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{base_total:,.0f}"
            )
        
        with col2:
            optimistic_diff = optimistic_total - base_total
            optimistic_pct = (optimistic_diff / base_total * 100) if base_total != 0 else 0
            st.metric(
                "Optimistic Scenario",
                f"${optimistic_total:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{optimistic_total:,.0f}",
                f"+{optimistic_pct:.1f}%"
            )
        
        with col3:
            pessimistic_diff = pessimistic_total - base_total
            pessimistic_pct = (pessimistic_diff / base_total * 100) if base_total != 0 else 0
            st.metric(
                "Pessimistic Scenario",
                f"${pessimistic_total:,.0f}" if metric.lower() in ['revenue', 'cost', 'profit'] else f"{pessimistic_total:,.0f}",
                f"{pessimistic_pct:.1f}%"
            )
        
        # Display scenario table
        st.markdown("##### 📋 Detailed Scenario Values")
        
        # Format the dataframe
        display_df = scenario_df.copy()
        if metric.lower() in ['revenue', 'cost', 'profit', 'price']:
            for col in ['Base Scenario', 'Optimistic Scenario', 'Pessimistic Scenario']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    def _advanced_settings(self, df: pd.DataFrame, date_col: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Advanced forecasting settings and diagnostics"""
        st.markdown("### ⚙️ Advanced Settings")
        
        results = {}
        
        st.markdown("#### 🔧 Model Parameters")
        
        # Prophet advanced settings
        if prophet_available:
            with st.expander("Prophet Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    changepoint_prior_scale = st.slider(
                        "Changepoint Prior Scale",
                        min_value=0.001,
                        max_value=0.5,
                        value=0.05,
                        step=0.001,
                        help="Flexibility of the trend"
                    )
                
                with col2:
                    seasonality_prior_scale = st.slider(
                        "Seasonality Prior Scale",
                        min_value=0.01,
                        max_value=10.0,
                        value=10.0,
                        step=0.1,
                        help="Strength of seasonality"
                    )
        
        # ARIMA advanced settings
        if statsmodels_available:
            with st.expander("ARIMA Settings"):
                st.markdown("**Automatic Parameter Selection**")
                
                if st.button("Auto-select ARIMA parameters"):
                    with st.spinner("Finding optimal ARIMA parameters..."):
                        # This would implement auto-ARIMA
                        st.info("Auto-ARIMA coming in future version")
        
        # Feature engineering
        st.markdown("#### 🛠️ Feature Engineering")
        
        include_features = st.multiselect(
            "Include additional features:",
            options=['Day of Week', 'Month', 'Quarter', 'Year', 'Is Weekend', 'Is Holiday'],
            default=['Month', 'Quarter']
        )
        
        st.markdown("#### 📊 Diagnostics")
        
        if st.button("Run Forecast Diagnostics", use_container_width=True):
            metric = st.selectbox(
                "Select metric for diagnostics:",
                options=numeric_cols,
                key='diagnostic_metric'
            )
            
            ts_data = self._prepare_time_series(df, date_col, metric)
            
            if not ts_data.empty:
                self._run_forecast_diagnostics(ts_data, metric)
        
        return results
    
    def _run_forecast_diagnostics(self, ts_data: pd.DataFrame, metric: str):
        """Run forecast diagnostics"""
        st.markdown("##### 📈 Forecast Diagnostics")
        
        # Residual analysis placeholder
        st.info("Forecast diagnostics and residual analysis coming in future version.")
        
        # Model persistence
        st.markdown("##### 💾 Model Persistence")
        
        if st.button("Save Current Model", use_container_width=True):
            st.success("Model save functionality coming soon")
        
        if st.button("Load Saved Model", use_container_width=True):
            st.info("Model load functionality coming soon")


# Streamlit integration function
def display_forecasting_section(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Main function to display forecasting interface
    """
    if not dataframes:
        st.warning("Please upload and analyze data first.")
        return {}
    
    forecaster = FinancialForecaster()
    forecast_results = forecaster.display_forecasting_interface(dataframes)
    
    return forecast_results