# # Use the official Python image from the Docker Hub
# FROM python:3.9-slim

# # Install manually all the missing libraries
# RUN apt-get update && apt-get install -y \
#     gconf-service \
#     libasound2 \
#     libatk1.0-0 \
#     libcairo2 \
#     libcups2 \
#     libfontconfig1 \
#     libgdk-pixbuf2.0-0 \
#     libgtk-3-0 \
#     libnspr4 \
#     libpango-1.0-0 \
#     libxss1 \
#     fonts-liberation \
#     libappindicator1 \
#     libnss3 \
#     lsb-release \
#     xdg-utils \
#     wget \
#     unzip

# # Install Chrome
# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# RUN dpkg -i google-chrome-stable_current_amd64.deb; apt-get -fy install


# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . .

# # Install pip dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Run search.py when the container launches
# CMD ["python", "search.py"]






# # FROM python:3.9-slim

# # # Set the working directory in the container
# # WORKDIR /app

# # # Install dependencies for running Chrome
# # RUN apt-get update && apt-get install -y \
# #     wget \
# #     gnupg \
# #     curl \
# #     google-chrome-stable \
# #     --no-install-recommends && \
# #     rm -rf /var/lib/apt/lists/*

# # # Install Chromedriver
# # RUN CHROMEDRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
# #     wget -N https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip -P /tmp && \
# #     unzip /tmp/chromedriver_linux64.zip -d /usr/local/bin && \
# #     rm /tmp/chromedriver_linux64.zip

# # # Copy the current directory contents into the container at /app
# # COPY . .

# # # Install pip dependencies
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Run search.py when the container launches
# # CMD ["python", "search.py"]



# First stage: Install Chrome and necessary libraries
FROM python:3.9-slim as build

# Install Chrome and necessary libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gconf-service \
    libasound2 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libfontconfig1 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libxss1 \
    fonts-liberation \
    libappindicator1 \
    libnss3 \
    lsb-release \
    xdg-utils \
    wget \
    unzip && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    dpkg -i google-chrome-stable_current_amd64.deb || apt-get -fy install && \
    rm google-chrome-stable_current_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Create the final image
FROM python:3.9-slim

# Copy the installed Chrome and Python dependencies from the build stage
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build /opt/google/chrome /opt/google/chrome

# Set environment variables for Chrome
ENV CHROME_BIN=/opt/google/chrome/google-chrome
ENV PATH=$PATH:/opt/google/chrome

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Run search.py when the container launches
CMD ["python", "search.py"]