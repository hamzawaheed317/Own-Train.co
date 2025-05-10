import React, { useState } from "react";

import "./AdminPannel.css";
import Dashboard from "./adminComponents/Dashboard";
import QueryRequests from "./adminComponents/QueryRequests";
import TrainModel from "./adminComponents/TrainModel";
import Feedbacks from "./adminComponents/Feedbacks";
import Sidebar from "./adminComponents/Sidebar";

const AdminPanel = () => {
  const [activeTab, setActiveTab] = useState("dashboard");

  const renderTab = () => {
    switch (activeTab) {
      case "dashboard":
        return <Dashboard />;
      // case "queries":
      //   return <QueryRequests />;
      case "train":
        return <TrainModel />;
      case "feedbacks":
        return <Feedbacks />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="admin-panel">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">{renderTab()}</main>
    </div>
  );
};

export default AdminPanel;
