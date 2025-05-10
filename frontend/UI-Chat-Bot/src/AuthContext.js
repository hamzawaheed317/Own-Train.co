// AuthContext.js
import { createContext, useContext, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [authState, setAuthState] = useState({
    isAuthenticated: false,
    isAdmin: false,
    admin: null,
    loading: true,
  });
  const navigate = useNavigate();

  const verifyAuth = async () => {
    try {
      const response = await fetch("http://localhost:5000/admin/verify-auth", {
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        setAuthState({
          isAuthenticated: true,
          isAdmin: true,
          admin: data.admin,
          loading: false,
        });
      } else {
        setAuthState({
          isAuthenticated: false,
          isAdmin: false,
          admin: null,
          loading: false,
        });
      }
    } catch (error) {
      console.error("Auth verification failed:", error);
      setAuthState({
        isAuthenticated: false,
        isAdmin: false,
        admin: null,
        loading: false,
      });
    }
  };

  useEffect(() => {
    verifyAuth();
  }, []);

  const value = {
    ...authState,
    verifyAuth,
  };

  return (
    <AuthContext.Provider value={value}>
      {!authState.loading && children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
