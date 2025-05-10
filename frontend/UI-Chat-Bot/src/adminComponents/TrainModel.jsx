import React, { useState, useRef, useEffect } from "react";
import "./TrainModel.css";

const TrainModel = () => {
  const [trainingMaterials, setTrainingMaterials] = useState([]);

  const [textInput, setTextInput] = useState("");
  const [showFileList, setShowFileList] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const handleTextChange = (e) => {
    setTextInput(e.target.value);
    autoResizeTextarea();
  };

  const autoResizeTextarea = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  };

  const handleFileUpload = (e) => {
    const newFiles = Array.from(e.target.files);
    setUploadedFiles((prevFiles) => {
      // Filter out duplicates by name
      const uniqueNewFiles = newFiles.filter(
        (newFile) =>
          !prevFiles.some((existingFile) => existingFile.name === newFiles.name)
      );
      return [...prevFiles, ...uniqueNewFiles];
    });
  };

  const removeUploadedFile = (index) => {
    console.log("Uploaded files", uploadedFiles);
    const newFiles = [...uploadedFiles];
    newFiles.splice(index, 1);
    setUploadedFiles(newFiles);
  };

  useEffect(() => {
    const fetchStoredFiles = async () => {
      try {
        const response = await fetch("http://localhost:5000/admin/files", {
          method: "GET",
          credentials: "include",
          headers: {
            Accept: "application/json",
          },
        });

        console.log("Full response:", response);

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.message || "Failed to fetch files");
        }

        const result = await response.json();
        console.log("Response data:", result);

        if (!result.success || !result.data) {
          throw new Error("Invalid response format from server");
        }

        // Transform the backend data to match your frontend structure
        const formattedFiles = result.data.map((file) => ({
          id: file._id,
          name: file.originalname || `File ${file._id}`,
          files: [
            {
              id: file._id,
              name: file.originalname,
              type:
                file.mimetype?.split("/")[1] ||
                file.originalname?.split(".").pop()?.toUpperCase() ||
                "FILE",
              fileObject: file,
            },
          ],
          text: "", // Add if you have text content
          selected: false,
          // trained: file.trained || false,
          trained: true,
        }));

        setTrainingMaterials(formattedFiles);
      } catch (error) {
        console.error("Error fetching files:", error);
        alert(error.message || "Failed to load files. Please try again.");
      }
    };

    fetchStoredFiles();
  }, []);

  const addTrainingMaterial = () => {
    if (uploadedFiles.length > 0 || textInput.trim()) {
      // Create text file if text exists
      let filesToUpload = [...uploadedFiles];
      let textFile = null;

      console.log(filesToUpload);
      if (textInput.trim()) {
        const textBlob = new Blob([textInput], { type: "text/plain" });
        textFile = new File([textBlob], "text-content.txt", {
          type: "text/plain",
        });
        filesToUpload.push(textFile);
      }
      console.log(filesToUpload);
      // Create form data for backend upload
      const formData = new FormData();
      filesToUpload.forEach((file) => {
        console.log(file);
        formData.append("files", file);
      });
      for (let [key, value] of formData.entries()) {
        console.log(key, value);
      }

      console.log("form data", formData);
      setIsTraining(true);
      // Send to backend
      fetch("http://localhost:5000/admin/upload-multiple", {
        method: "POST",
        credentials: "include", // This sends cookies automatically
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("data", data);
          if (data) {
            // Create new material after successful upload
            console.log("files To Upload", filesToUpload),
              console.log("uploaded files", uploadedFiles);
            const newMaterial = {
              id: Date.now(),
              name: `Training Set ${trainingMaterials.length + 1}`,
              files: filesToUpload.map((file, index) => ({
                id: data.fileIds[index], //yeh wo id hai jo mongo sy mujhe mili hai
                name: file.name,
                type: file.name.split(".").pop().toUpperCase(),
                fileObject: file,
              })),
              text: textInput.trim(),
              selected: true,
              trained: true,
            };
            console.log(newMaterial);

            setIsTraining(false);
            setTrainingMaterials([...trainingMaterials, newMaterial]);
            setUploadedFiles([]);
            setTextInput("");

            alert(`${data.count} Dataset Upload Successfully`);
          } else {
            alert("Failed to upload files to server");
          }
        })
        .catch((error) => {
          console.error("Upload error:", error);
          setIsTraining(false);
          alert(`Error Uploading Files! Please Enter Valid Files `);
        });
    }
  };

  const removeTrainingMaterial = (id) => {
    setTrainingMaterials(trainingMaterials.filter((mat) => mat.id !== id));
  };

  const toggleMaterialSelection = (id) => {
    setTrainingMaterials(
      trainingMaterials.map((mat) =>
        mat.id === id ? { ...mat, selected: !mat.selected } : mat
      )
    );
  };

  const untrainMaterial = async (id) => {
    console.log("Trained Materials", trainingMaterials);

    //dataset jis ko untrain krna hai usko find krna , us mein jitni bhi files hain un sb ki ids honi chhaiye
    const dataSetToDel = trainingMaterials.find((mat) => {
      return mat.id === id;
    });

    //data set mil gaya
    console.log(dataSetToDel);
    console.log(dataSetToDel.files);

    //ab array banani hai file id ki jin ko del krna hia backend pay request bjhny kay liay
    // Get all file IDs from the material's files array
    const fileIdsToDelete = dataSetToDel.files.map((file) => file.id);
    console.log(fileIdsToDelete);

    // Make API call to delete files
    const deleteResponse = await fetch("http://localhost:5000/admin/files", {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify({ fileIds: fileIdsToDelete }),
    });

    console.log("Response", deleteResponse);
    if (!deleteResponse.ok) {
      throw new Error("Failed to delete files from server");
    }

    //updating the state of the training materials , pehly materials wesy hi rehny do ,and jis pay click kiya hia unki train state ko false krdo
    const finalMaterials = trainingMaterials.map((mat) =>
      mat.id === id ? { ...mat, trained: false } : mat
    );
    setTrainingMaterials(finalMaterials);

    console.log("finalMaterials", finalMaterials);
    alert(
      `Model is untrained from ${
        trainingMaterials.find((m) => m.id === id).name
      }`
    );
  };

  const trainModel = () => {
    const selectedMaterials = trainingMaterials.filter((mat) => mat.selected);
    if (selectedMaterials.length === 0) {
      alert("Please select training materials to train");
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          setTrainingMaterials(
            trainingMaterials.map((mat) =>
              mat.selected ? { ...mat, trained: true } : mat
            )
          );
          return 0;
        }
        return prev + 10;
      });
    }, 300);
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        trainModel();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [trainingMaterials]);

  return (
    <div className="train-model-container">
      <div className="chat-gpt-style">
        <div className="training-content">
          <h1>Train Your AI Model</h1>
          <p className="subtitle">
            Give files to train your model (e.g: .docx, .pdf , .txt files and
            raw text)
          </p>

          <div className="text-input-container">
            {/* Uploaded files preview */}
            {uploadedFiles.length > 0 && (
              <div className="uploaded-files-preview">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="uploaded-file-item">
                    <span className="file-name">{file.name}</span>
                    <button
                      className="remove-file-btn"
                      onClick={() => removeUploadedFile(index)}
                    >
                      x
                    </button>
                  </div>
                ))}
              </div>
            )}

            <textarea
              ref={textareaRef}
              value={textInput}
              onChange={handleTextChange}
              placeholder="Enter training text here (will be combined with any uploaded files)..."
              rows={3}
            />

            <div className="input-actions">
              <button
                className="attach-btn"
                onClick={() => fileInputRef.current.click()}
              >
                <svg viewBox="0 0 24 24" className="attach-icon">
                  <path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z" />
                </svg>
                Add Files
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".pdf,.doc,.docx,.txt"
                multiple
                hidden
              />

              <button
                className={`train-btn ${isTraining ? "training" : ""}`}
                onClick={addTrainingMaterial}
                disabled={isTraining}
              >
                {isTraining ? (
                  <>
                    <span className="spinner"></span>
                    Training... {trainingProgress}%
                  </>
                ) : (
                  "Train Model"
                )}
              </button>
            </div>
          </div>

          <div className="file-list-toggle">
            <button onClick={() => setShowFileList(!showFileList)}>
              {showFileList ? "Hide" : "Show"} Training Materials (
              {trainingMaterials.length})
            </button>
          </div>

          {showFileList && (
            <div className="file-list-container">
              <div className="file-list-header">
                <h3>Training Materials</h3>
                <div className="header-stats">
                  <span className="selected-count">
                    {trainingMaterials.filter((f) => f.selected).length}{" "}
                    selected
                  </span>
                  <span className="total-count">
                    {trainingMaterials.length} total
                  </span>
                </div>
              </div>

              {trainingMaterials.length === 0 ? (
                <div className="empty-state">
                  <p>No training materials added yet</p>
                </div>
              ) : (
                <ul className="file-list">
                  {trainingMaterials.map((material) => (
                    <li
                      key={material.id}
                      className={`material-item ${
                        material.selected ? "selected" : ""
                      } ${material.trained ? "trained" : ""}`}
                    >
                      <div className="material-main">
                        <input
                          type="checkbox"
                          checked={material.selected}
                          onChange={() => toggleMaterialSelection(material.id)}
                          className="material-checkbox"
                        />
                        <div className="material-content">
                          <div className="material-header">
                            <h4 className="material-name">{material.name}</h4>
                            {material.trained && (
                              <span className="trained-indicator">
                                <span className="trained-dot"></span>
                                Trained
                              </span>
                            )}
                          </div>

                          <div className="material-details">
                            {material.files.length > 0 && (
                              <div className="files-section">
                                <span className="section-label">Files:</span>
                                <div className="file-tags">
                                  {material.files.map((file) => (
                                    <span key={file.id} className="file-tag">
                                      {file.name}
                                      <span className="file-type">
                                        {file.type}
                                      </span>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {material.text && (
                              <div className="text-section">
                                <span className="section-label">Text:</span>
                                <p className="text-preview">
                                  {material.text.length > 50
                                    ? `${material.text.substring(0, 50)}...`
                                    : material.text}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="material-actions">
                        {material.trained ? (
                          <button
                            className="btn untrain-btn"
                            onClick={() => untrainMaterial(material.id)}
                          >
                            Untrain
                          </button>
                        ) : (
                          <span className="not-trained-label">
                            Not trained yet
                          </span>
                        )}
                        {/* <button
                          className="btn remove-btn"
                          onClick={() => removeTrainingMaterial(material.id)}
                        >
                          Remove
                        </button> */}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainModel;
