import React from 'react';
import './style.css';

function StockDataDisplay({ data, error, stockName }) {
    if (error) {
        return <p className="error-message">{error}</p>;
    }

    if (!data) {
        return <p className="info-message">Enter stock name and date to fetch data</p>;
    }

    return (
        <div className="data-container">
            <h2>{stockName}</h2>
            <img src={`data:image/png;base64,${data.main_image}`} alt="Stock Price Chart" />
            <img src={`data:image/png;base64,${data.future_image}`} alt="Zoomed" />
        </div>
    );
}

export default StockDataDisplay;
