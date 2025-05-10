import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

// ProtectedRouteUser.jsx
const ProtectedRouteUser = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const userJson = localStorage.getItem("user");
        if (!userJson) {
          throw new Error("No user data");
        }

        // Safe JSON parsing
        const user = JSON.parse(userJson);
        if (!user?.id) {
          throw new Error("Invalid user data");
        }

        setIsAuthenticated(true);
      } catch (error) {
        console.error("Auth check failed:", error);
        navigate("/login");
      }
    };

    checkAuth();
  }, []);

  if (isAuthenticated === null) {
    return <div></div>; // Or your loading component
  }

  return isAuthenticated ? children : null;
};

export default ProtectedRouteUser;
