import React, { useState } from "react";
import axios from "axios";
import styles from "./FeedbackWidget.module.css";

const FeedbackButtons = ({ responseId }) => {
  //ider response id kay pas meesage id hai
  const [showReasons, setShowReasons] = useState(false);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const reasons = [
    "Inaccurate information",
    "Not helpful",
    "Too verbose",
    "Too brief",
    "Other",
  ];

  const submitFeedback = async (type, reason = "") => {
    console.log("Submit berofe", responseId, type, reason);
    const user = localStorage.getItem("user");
    console.log("User form session", user);
    try {
      console.log();
      const response = await fetch("http://localhost:5000/feedback/post", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          responseId,
          feedbackType: type,
          reason,
          user,
        }),
      });

      console.log(response);
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setFeedbackSubmitted(true);
      return data;
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  if (feedbackSubmitted) {
    setTimeout(() => {
      setFeedbackSubmitted(false);
    }, 5000);
    return (
      <div className={styles["feedback-thankyou"]}>
        Thank you for your feedback!
      </div>
    );
  }

  return (
    <div className={styles["feedback-container"]}>
      <div className={styles["feedback-buttons"]}>
        <button
          className={`${styles["feedback-btn"]} ${styles["like-btn"]}`}
          onClick={() => submitFeedback("like")}
        >
          <span role="img" aria-label="like">
            👍
          </span>{" "}
        </button>

        <button
          className={`${styles["feedback-btn"]} ${styles["dislike-btn"]}`}
          onClick={() => setShowReasons(true)}
        >
          <span role="img" aria-label="dislike">
            👎
          </span>{" "}
        </button>
      </div>

      {showReasons && (
        <div className={styles["feedback-reasons"]}>
          <p>What was wrong with this response?</p>
          {reasons.map((reason) => (
            <button
              key={reason}
              className={styles["reason-btn"]}
              onClick={() => submitFeedback("dislike", reason)}
            >
              {reason}
            </button>
          ))}
          <button
            className={styles["cancel-btn"]}
            onClick={() => setShowReasons(false)}
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};

export default FeedbackButtons;
