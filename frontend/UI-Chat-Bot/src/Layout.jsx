import { Outlet } from "react-router-dom";
import LandingPage from "./LandingPage";

const Layout = () => {
  return (
    <>
      <LandingPage />
      <Outlet /> {/* This will render child routes */}
    </>
  );
};

export default Layout;
