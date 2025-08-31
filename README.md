# Project Setup Guide

This project consists of a Next.js frontend and a Python/Uvicorn backend.

## Frontend Setup (Next.js)

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

### Getting Started

First, install the frontend dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
# or
bun install
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Backend Setup (Python/Uvicorn)

The backend is built with Python and served using Uvicorn.

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Setup Steps

1.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

2.  **Create a Python virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    - **On Windows:**
      ```bash
      .\venv\Scripts\activate
      ```
    - **On macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```

4.  **Install backend dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Uvicorn server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend server will typically run on [http://127.0.0.1:8000](http://127.0.0.1:8000) or a similar address.

## Killing Backend Processes

If you need to stop the Uvicorn server or other backend processes that might be running in the background, you can use the following methods:

### On Windows

1.  **Find the process ID (PID) of the Uvicorn process:**
    Open Command Prompt or PowerShell and run:

    ```bash
    netstat -ano | findstr :8000
    ```

    (Replace `8000` with the actual port if your Uvicorn server is running on a different port).
    Look for the line that shows `LISTENING` and note the PID (the last number in the line).

2.  **Terminate the process using its PID:**
    ```bash
    taskkill /PID [PID] /F
    ```
    (Replace `[PID]` with the actual PID you found).

### On macOS/Linux

1.  **Find the process ID (PID) of the Uvicorn process:**
    Open your terminal and run:

    ```bash
    lsof -i :8000
    ```

    (Replace `8000` with the actual port if your Uvicorn server is running on a different port).
    Look for the `PID` column.

2.  **Terminate the process using its PID:**
    ```bash
    kill -9 [PID]
    ```
    (Replace `[PID]` with the actual PID you found).

### General Method (if you know the process name)

You can also try to kill processes by name, though this can be less precise:

- **On Windows:**

  ```bash
  taskkill /IM uvicorn.exe /F
  ```

  (This will kill all processes named `uvicorn.exe`. Use with caution).

- **On macOS/Linux:**
  ```bash
  pkill uvicorn
  ```
  (This will kill all processes with "uvicorn" in their name. Use with caution).

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
