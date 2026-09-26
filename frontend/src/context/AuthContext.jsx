import React, { createContext, useContext, useState, useEffect } from "react";
import { api } from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => api.getSavedUser());
  const [token, setToken] = useState(() => localStorage.getItem("fedmed_access_token"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Validate existing token on mount
    async function verifyAuth() {
      if (token) {
        try {
          const profile = await api.getMe();
          setUser(profile);
          localStorage.setItem("fedmed_user", JSON.stringify(profile));
        } catch {
          // Token invalid or expired
          api.logout();
          setUser(null);
          setToken(null);
        }
      }
      setLoading(false);
    }
    verifyAuth();
  }, [token]);

  const login = async (username, password) => {
    const res = await api.login(username, password);
    setToken(res.access_token);
    setUser(res.user);
    return res.user;
  };

  const register = async (payload) => {
    const res = await api.register(payload);
    setToken(res.access_token);
    setUser(res.user);
    return res.user;
  };

  const logout = () => {
    api.logout();
    setToken(null);
    setUser(null);
  };

  const value = {
    user,
    token,
    role: user?.role || "GUEST",
    isAuthenticated: !!user,
    loading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
