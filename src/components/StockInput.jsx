import React from 'react';

function StockInput({ stockName, setStockName, endDate, setEndDate, fetchStockData, buttonClicked }) {
    return (
        <div>
            <input
                type="text"
                placeholder="Enter stock name"
                value={stockName}
                onChange={(e) => setStockName(e.target.value)}
            />
            <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
            />
            <button onClick={fetchStockData} disabled={buttonClicked}>
                Fetch Data
            </button>
        </div>
    );
}

export default StockInput;
