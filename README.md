
# PostgreSQL Installation Guide

This guide provides step-by-step instructions to install PostgreSQL on various operating systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation on Ubuntu](#installation-on-ubuntu)
- [Installation on macOS](#installation-on-macos)
- [Post-Installation Setup](#post-installation-setup)
- [Troubleshooting](#troubleshooting)
- [PgAi Extension for PostgreSQL](#pgai-extension-for-postgresql)
- [Contributing](#contributing)
- [Contact](#contact)

## Prerequisites

- Ensure you have administrative privileges on your system.
- An internet connection to download necessary files.

## Installation on Ubuntu

1. Update the package list:
    ```sh
    sudo apt update
    ```

2. Install PostgreSQL:
    ```sh
    sudo apt install postgresql postgresql-contrib
    ```

3. Verify the installation:
    ```sh
    psql --version
    ```

## Installation on macOS

1. Install Homebrew if you haven't already:
    ```sh
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. Install PostgreSQL using Homebrew:
    ```sh
    brew install postgresql
    ```

3. Start the PostgreSQL service:
    ```sh
    brew services start postgresql
    ```

4. Verify the installation:
    ```sh
    psql --version
    ```

## Post-Installation Setup

### Creating a New User

1. Switch to the `postgres` user:
    ```sh
    sudo -i -u postgres
    ```

2. Open the PostgreSQL prompt:
    ```sh
    psql
    ```

3. Create a new user:
    ```sql
    CREATE USER yourusername WITH PASSWORD 'yourpassword';
    ```

4. Grant privileges to the new user:
    ```sql
    ALTER USER yourusername WITH SUPERUSER;
    ```

5. Exit the PostgreSQL prompt:
    ```sh
    \q
    ```

### Creating a New Database

1. Switch to the `postgres` user if not already done:
    ```sh
    sudo -i -u postgres
    ```

2. Create a new database:
    ```sh
    createdb yourdatabase
    ```

## Troubleshooting

### Common Issues

- **Unable to connect to the server:** Ensure the PostgreSQL server is running and listening on the correct port.
- **Authentication failed:** Verify the username and password are correct.
- **Permission denied:** Ensure you have the necessary privileges to perform the desired action.

For more detailed troubleshooting, refer to the [PostgreSQL documentation](https://www.postgresql.org/docs/).

---

Feel free to contribute to this guide by submitting a pull request or opening an issue on our GitHub repository.

---

# PgAi Extension for PostgreSQL

## Introduction

**Postgres - AI Extension (PgAi):** PgAi integrates predictive analytics directly into PostgreSQL, allowing users to execute "predictive queries" seamlessly. This extension brings the power of AI-driven predictions into the database, enabling data-driven decisions and advanced analytics without the need for external tools.

## Installation Instructions

1. **Prerequisites:**  
   Ensure you have PostgreSQL installed. Download from the official [PostgreSQL website](https://www.postgresql.org/download/).

2. **Cloning the Repository:**  
   Clone the pgai repository and navigate to the directory:
   ```bash
   git clone git@github.com:moiz697/PgAi.git
   cd pgai
   ```

3. **Create .env File:**  
   Create a `.env` file in the root directory of your project with the following content:
   ```env
   PORT=5432
   USERNAME=yourusername
   DATABASE=yourdatabase
   PASSWORD=yourpassword
   ```

4. **Export PostgreSQL Path:**  
   Export the PostgreSQL bin directory to your PATH. Replace `/path/to/postgres` with your PostgreSQL installation path:
   ```bash
   export PATH=/path/to/postgres/bin:$PATH
   ```

5. **Build and Install the Extension:**
   ```bash
   make && make install
   ```

6. **Run PostgreSQL:**  
   Ensure PostgreSQL is running:
   ```bash
   pg_ctl start
   ```

7. **Create the Extension:**  
   Connect to your PostgreSQL instance and create the pgai extension:
   ```sql
   CREATE EXTENSION pgai;
   ```

## Stock Data Instructions

1. **Creating the Stock Table:**  
   Create the `EXAMPLE_stock` table to store stock data:
   ```sql
   CREATE TABLE IF NOT EXISTS EXAMPLE_stock (
       date DATE PRIMARY KEY,
       open DOUBLE PRECISION,
       high DOUBLE PRECISION,
       low DOUBLE PRECISION,
       close DOUBLE PRECISION,
       adj_close DOUBLE PRECISION,
       volume BIGINT
   );
   ```

2. **Downloading Stocks from Yahoo Finance:**  
   Visit [Yahoo Finance](https://finance.yahoo.com) to download stock data.

3. **Importing Stock Data into PostgreSQL:**  
   Use the `COPY` command to import the downloaded stock data into the PostgreSQL table:
   ```sql
   COPY EXAMPLE_stock(date, open, high, low, close, adj_close, volume)
   FROM '/path/to/example.csv'
   DELIMITER ','
   CSV HEADER;
   ```

## Choosing a Model

We provide four models:
- LSTM
- Prophet
- ARIMA
- SARIMA

## Configuration

1. **Add Configurations in PostgreSQL:**  
   Create a table to store database connection details:
   ```sql
   CREATE TABLE IF NOT EXISTS db_config (
       key TEXT PRIMARY KEY,
       value TEXT NOT NULL
   );

   INSERT INTO db_config (key, value) VALUES
       ('db_host', 'localhost'),
       ('db_port', '5432'),
       ('db_name', 'yourdatabase'),
       ('db_user', 'yourusername'),
       ('db_password', 'yourpassword');
   ```

You can also write your own model and make changes in the `pgai--1.0` file.

## Usage

To use the pgai extension, follow these steps:

1. **Load the Extension:**
   ```sql
   CREATE EXTENSION pgai;
   ```

2. **Run Predictive Queries:**
   Use the provided models to perform predictive analytics directly within your PostgreSQL database. Example query:
   ```sql
   SELECT * FROM EXAMPLE_stock('2025-09-21');
   ```

Refer to the `pgai` documentation for detailed examples and usage patterns.

## Pseudo Columns and Applications

The PgAi extension allows the integration of pseudo columns that fetch predictive values from trained models. While the primary example provided is stock data prediction, the same methodology can be applied to other datasets with minimal modifications. For instance, you can use PgAi for:

- **Weather Forecasting:** Predict future weather conditions based on historical data.
- **Sales Projections:** Forecast future sales using past sales data.
- **Resource Allocation:** Predict future resource needs in logistics or supply chain management.

By adjusting the models and the SQL functions, you can adapt the PgAi extension to various domains. For any help regarding this, feel free to reach out to the team. We are happy to help you.

## Why PgAi is Better

1. **Integrated Predictive Analytics:** PgAi brings predictive analytics directly into PostgreSQL, eliminating the need for external tools or platforms. This seamless integration simplifies the workflow and enhances performance.
2. **Flexibility:** The extension is designed to be adaptable to various types of data. Whether it's stock prices, weather forecasting, or sales projections, PgAi can be customized to meet specific requirements.
3. **Efficiency:** By running predictive queries within the database, PgAi reduces data transfer overhead and speeds up the analysis process.
4. **Ease of Use:** With simple SQL commands, users can perform complex predictive analytics, making advanced data science techniques accessible to a broader audience.

## Contributing

We welcome contributions. Fork the repository and submit pull requests.

## Contact

For any questions or feedback, please open an issue on the GitHub repository or contact us at pgartificialintelligence@gmail.com.

**LinkedIn Profiles:**
- [Abdul Moiez Ibrar](https://www.linkedin.com/in/abdul-moiez-ibrar-79167b104/)
- [Mustafa Khattak](https://www.linkedin.com/in/mustafa-khattak/)
- [Umer Khurshid](https://www.linkedin.com/in/umer-khurshid-b1a815271/)

---

Happy Predicting with pgai!
