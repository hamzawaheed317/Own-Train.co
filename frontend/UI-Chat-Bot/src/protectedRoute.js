// ProtectedRoute.js
import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext";

const ProtectedRoute = ({ children, adminOnly = false }) => {
  const { isAuthenticated, isAdmin, loading } = useAuth();

  if (loading) return <div></div>;

  if (!isAuthenticated) {
    return <Navigate to="/admin/login" replace />;
  }

  if (adminOnly && !isAdmin) {
    return <Navigate to="/admin-pannel" replace />;
  }

  return children;
};

export default ProtectedRoute;
