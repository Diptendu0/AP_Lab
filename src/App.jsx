import React, { useState } from 'react';
import axios from 'axios';
import StockInput from './components/StockInput';
import StockDataDisplay from './components/StockDataDisplay';
import './index.css';

function App() {
  const [stockName, setStockName] = useState("");
  const [endDate, setEndDate] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [buttonClicked, setButtonClicked] = useState(false);

  const fetchStockData = async () => {
    if (stockName && endDate) {
      setButtonClicked(true);
      try {
        const response = await axios.get(`http://127.0.0.1:5000/api/stock?name=${stockName}&end_date=${endDate}`);
        setData(response.data);
        setError(null);
      } catch (err) {
        setError(err.response?.data?.error || "Error fetching data");
        setData(null);
      } finally {
        setTimeout(() => setButtonClicked(false), 200);
      }
    }
  };

  return (
    <div className="app-container">
      <h1>Stock Price Viewer</h1>
      <div className="input-container">
        <StockInput
          stockName={stockName}
          setStockName={setStockName}
          endDate={endDate}
          setEndDate={setEndDate}
          fetchStockData={fetchStockData}
          buttonClicked={buttonClicked}
        />
        <StockDataDisplay data={data} error={error} stockName={stockName}/>
      </div>
    </div>
  );
}

export default App;