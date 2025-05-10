//this is code for register admin
const jwt = require("jsonwebtoken");
const Admin = require("../models/admin");

const registerAdmin = async (req, res) => {
  console.log("register adming called");
  try {
    console.log("Register admin called ");
    const {
      clientname,
      email,
      password,
      companyname,
      companyurl,
      country,
      region,
      telephone,
    } = req.body;

    console.log(
      clientname,
      email,
      password,
      companyname,
      companyurl,
      country,
      region,
      telephone
    );
    if (
      !clientname ||
      !email ||
      !password ||
      !companyname ||
      !companyurl ||
      !country ||
      !region ||
      !telephone
    ) {
      return res.status(400).json({ message: "All fields are required" });
    }

    const existingAdmin = await Admin.findOne({ email });
    if (existingAdmin) {
      return res.status(409).json({ message: "Admin already exists" });
    }

    const newAdmin = new Admin({
      clientname,
      email,
      password,
      companyname,
      companyurl,
      country,
      region,
      telephone,
    });
    await newAdmin.save();

    res.status(201).json({
      message: "Admin registered successfully",
      admin: {
        id: newAdmin._id,
        clientname: newAdmin.clientname,
        email: newAdmin.email,
        companyName: newAdmin.companyname,
        companyUrl: newAdmin.companyurl,
        country: newAdmin.country,
        region: newAdmin.region,
        telephone: newAdmin.telephone,
        role: newAdmin.role,
        createdAt: newAdmin.createdAt,
      },
    });
  } catch (error) {
    console.error("Error in registerAdmin:", error);
    res.status(500).json({ message: "Server error" });
  }
};

module.exports = registerAdmin;
