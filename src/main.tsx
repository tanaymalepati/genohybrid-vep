import React from 'react';
import ReactDOM from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import App from './App';
import PredictPage from './pages/PredictPage';
import BatchPage from './pages/BatchPage';
import PerformancePage from './pages/PerformancePage';
import HistoryPage from './pages/HistoryPage';
import './styles.css';

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <PredictPage /> },
      { path: 'batch', element: <BatchPage /> },
      { path: 'performance', element: <PerformancePage /> },
      { path: 'history', element: <HistoryPage /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
