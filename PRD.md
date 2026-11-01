# PRODUCT REQUIREMENTS DOCUMENT (PRD)
# Smart Cafe AI – Intelligent Cafe Monitoring & Analytics System

**Document Version:** 1.0  
**Project Name:** Smart Cafe AI  
**Document Type:** Product Requirements Document  
**Project Status:** Prototype / Active Development  
**Platform:** Web Application + AI Computer Vision Engine  
**Primary Environment:** Windows Development Environment  
**Target Deployment:** Local Computer / On-Premises Server / Cloud (Future)

---

# 1. Project Overview

## 1.1 Introduction

Smart Cafe AI is an AI-powered cafe monitoring and analytics platform that uses computer vision and machine learning to understand activities happening inside a cafe.

The system uses CCTV cameras or webcams to detect people, identify objects, track customers and employees, analyze customer seating activities, and generate useful business insights.

The main objective is to help cafe owners understand how customers use their space, how long they occupy tables, which tables are available, and how efficiently the cafe is operating.

Instead of relying entirely on manual observation, Smart Cafe AI automatically processes camera footage and converts it into meaningful operational information.

## 1.2 Problem Statement

Cafe owners and managers often face challenges such as:

- They cannot easily determine how many customers are currently inside the cafe.
- They do not know which tables are occupied or available.
- They cannot accurately measure how long customers occupy tables.
- They have limited information about customer arrival and departure patterns.
- They cannot easily analyze seating utilization during different hours.
- Employee activities require manual supervision.
- Reviewing hours of CCTV recordings to understand cafe operations is time-consuming.
- Historical CCTV footage does not automatically provide useful business analytics.

Smart Cafe AI aims to solve these problems using real-time object detection, person tracking, activity analysis, and automated reporting.

## 1.3 Product Vision

To develop an intelligent cafe management and monitoring system that converts CCTV camera footage into actionable business insights.

The system should eventually provide cafe owners with information such as:

- Current customer count.
- Customer arrival and departure times.
- Customer seating duration.
- Table occupancy status.
- Available and occupied tables.
- Customer activity patterns.
- Employee activity insights.
- Peak cafe hours.
- Table utilization percentage.
- Historical operational reports.

---

# 2. Project Objectives

The primary objectives are:

1. Develop a real-time computer vision system using CCTV or webcam input.
2. Detect people and relevant cafe objects.
3. Track individuals across consecutive video frames.
4. Assign temporary tracking IDs to detected people.
5. Analyze whether a person is sitting or standing.
6. Identify customer seating events.
7. Calculate how long customers occupy seats or tables.
8. Detect when customers arrive and leave.
9. Identify occupied and available tables.
10. Store event information for historical analysis.
11. Develop a web dashboard for cafe monitoring.
12. Generate reports and operational analytics.
13. Maintain privacy by avoiding facial recognition and unnecessary personal identification.

---

# 3. Target Users

## 3.1 Cafe Owner

The cafe owner is the primary user.

Responsibilities:
- Monitor current cafe activity.
- Check customer count.
- View occupied and available tables.
- Analyze seating duration.
- View daily and monthly reports.
- Understand cafe utilization.

## 3.2 Cafe Manager

Responsibilities:
- Monitor customer activities.
- Observe seating patterns.
- Review employee activity insights.
- Check operational reports.
- Identify unusual operational patterns.

## 3.3 System Administrator

Responsibilities:
- Configure cameras.
- Manage the AI system.
- Configure camera settings.
- Monitor system health.
- Manage users and permissions.
- Maintain historical data.

---

# 4. Project Scope

The project will be developed incrementally.

## 4.1 Phase 1 – AI Computer Vision Engine

The first phase focuses on processing camera footage.

Features:

- Webcam/CCTV input.
- Real-time video processing.
- Person detection.
- Cafe object detection.
- Person tracking.
- Temporary person IDs.
- Pose estimation.
- Basic sitting/standing classification.
- Live detection visualization.

## 4.2 Phase 2 – Customer Seating Analytics

Features:

- Customer seating detection.
- Seat/table association.
- Seating start time.
- Seating end time.
- Customer seating duration.
- Occupied seat detection.
- Available seat detection.
- Customer arrival and departure events.
- Seating event storage.

## 4.3 Phase 3 – Backend and Database

Features:

- Backend API.
- Camera management.
- Detection event management.
- Customer activity event management.
- Seating event management.
- Analytics APIs.
- Historical data retrieval.

## 4.4 Phase 4 – Web Dashboard

Features:

- Dashboard overview.
- Live camera monitoring.
- Current customer count.
- Table occupancy overview.
- Seating duration analytics.
- Historical reports.
- Charts and visualizations.
- Camera and system settings.

## 4.5 Phase 5 – Advanced Analytics

Future features:

- Employee activity analysis.
- Customer behavior analysis.
- Peak-hour analysis.
- Table turnover analysis.
- Average customer visit duration.
- Operational efficiency insights.
- Automated alerts.
- Multiple camera support.
- Multiple cafe/location support.

---

# 5. Functional Requirements

## 5.1 Camera Input Management

**Requirement ID: FR-001**

The system must support video input from cameras.

Expected functionality:

- Connect to a local webcam.
- Support CCTV/IP camera streams.
- Capture continuous video frames.
- Check whether the camera is available.
- Handle camera disconnection.
- Display camera connection status.
- Allow camera configuration.
- Support multiple cameras in future versions.

Initial implementation:

OpenCV VideoCapture.

Future enhancement:

Support RTSP camera streams.

## 5.2 Object Detection

**Requirement ID: FR-002**

The system must detect relevant objects inside the cafe.

Initial object classes:

- Person
- Bottle
- Cup
- Chair
- Dining table
- Laptop
- Cell phone

Technology:

- YOLO11 object detection model.
- Ultralytics.
- PyTorch.
- OpenCV.

Expected output:

- Object class.
- Confidence score.
- Bounding box coordinates.
- Frame timestamp.

## 5.3 Person Detection

**Requirement ID: FR-003**

The system must identify people appearing in camera footage.

Expected functionality:

- Detect people in each frame.
- Generate bounding boxes.
- Calculate detection confidence.
- Provide person coordinates.
- Pass detections to the tracking module.

Person detection must not identify a real person's name or identity.

## 5.4 Person Tracking

**Requirement ID: FR-004**

The system must track detected people across consecutive video frames.

The system should assign a temporary tracking ID to every successfully tracked person.

Important:

A tracking ID is temporary and is not a permanent customer identity.

Expected functionality:

- Assign tracking IDs.
- Maintain IDs across frames.
- Handle short-term occlusion.
- Handle temporary detection loss.
- Re-establish tracks when possible.
- Remove expired tracks.
- Maintain tracking history.

The UI should distinguish:

- Tracked person: ID 1.
- Detected but untracked person: Unknown.

Unknown detections should not automatically create customer visits or seating records.

## 5.5 Human Pose Estimation

**Requirement ID: FR-005**

The system must estimate human body keypoints.

Technology:

YOLO11 Pose model (`yolo11n-pose.pt`).

Expected keypoints include COCO 17 keypoints.

## 5.6 Sitting and Standing Detection

**Requirement ID: FR-006**

The system must classify a person's posture:
- Sitting
- Standing
- Unknown

Must consider:
- Hip, knee, ankle positions
- Upper-body orientation & bounding box aspect ratio
- Temporal consistency smoothing
- Seat/chair spatial overlap

## 5.7 Seat and Table Management

**Requirement ID: FR-007**

The system must understand the seating arrangement of the cafe.
Each table has a unique ID and occupancy status (`Available`, `Occupied`, `Unknown`).

## 5.8 Customer Seating Event Detection

**Requirement ID: FR-008**

Create seating events when a person is reliably associated with a seat and posture remains seated stably.

## 5.9 Customer Departure Detection

**Requirement ID: FR-009**

Identify customer departure with configurable grace period to prevent false departures caused by temporary occlusion.

## 5.10 Seating Duration Calculation

**Requirement ID: FR-010**

Track active duration and record completed session duration.

## 5.11 Customer Count

**Requirement ID: FR-011**

Distinguish visible detections, active tracks, and seated customers.

## 5.12 Employee Activity Analytics (Future)

**Requirement ID: FR-012**

Designated operational zones (kitchen/counter) without facial identification.

---

# 6. Suggested Architecture & Roadmap

Refer to the PRD specification for the modular architecture:
- `ai-engine/` (camera, detection, tracking, activity, seating, events, config)
- `backend/` (FastAPI, SQLAlchemy, Pydantic, SQLite / PostgreSQL)
- `frontend/` (React / Next.js Dashboard)
