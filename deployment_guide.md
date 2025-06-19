# 🚀 Deployment Guide

This guide covers multiple deployment options for the BRCA Cancer Prediction System.

## 🌟 Quick Start (Streamlit Cloud) - Recommended

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io/cloud))

### Steps
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Select your repository and branch
   - Set main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live in minutes!**

---

## 🐳 Docker Deployment

### Local Docker
```bash
# Build the image
docker build -t brca-prediction .

# Run the container
docker run -p 8501:8501 brca-prediction

# Access at http://localhost:8501
```

### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  brca-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## ☁️ Cloud Platform Deployments

### 1. Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Steps
```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open your app
heroku open
```

#### Environment Variables (if needed)
```bash
heroku config:set VARIABLE_NAME=value
```

### 2. Railway Deployment

1. Connect your GitHub repository to [Railway](https://railway.app)
2. Select your repository
3. Railway will automatically detect the Python app and deploy
4. Your app will be available at the provided URL

### 3. Render Deployment

1. Go to [Render](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy!

---

## 🔧 Advanced Deployments

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
- Choose Ubuntu 20.04 LTS
- Select t2.micro (free tier eligible)
- Configure security group to allow HTTP (port 80) and custom port 8501

#### 2. Setup on EC2
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip git -y

# Clone your repository
git clone https://github.com/yourusername/brca-prediction.git
cd brca-prediction

# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

#### 3. Run as a Service (Optional)
Create `/etc/systemd/system/brca-app.service`:
```ini
[Unit]
Description=BRCA Prediction App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/brca-prediction
ExecStart=/usr/local/bin/streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable brca-app
sudo systemctl start brca-app
```

### Google Cloud Platform (GCP) Deployment

#### Using Google Cloud Run
1. **Build and push Docker image**:
   ```bash
   # Configure gcloud
   gcloud config set project YOUR_PROJECT_ID
   
   # Build and push to Container Registry
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/brca-prediction
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy brca-prediction \
     --image gcr.io/YOUR_PROJECT_ID/brca-prediction \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501
   ```

### Azure Container Instances

```bash
# Create resource group
az group create --name brca-rg --location eastus

# Deploy container
az container create \
  --resource-group brca-rg \
  --name brca-prediction \
  --image your-dockerhub-username/brca-prediction \
  --ports 8501 \
  --dns-name-label brca-prediction-unique
```

---

## 🔒 Security Considerations

### Environment Variables
Never commit sensitive data. Use environment variables:

```python
import os
SECRET_KEY = os.getenv('SECRET_KEY', 'default-value')
```

### HTTPS Configuration
For production deployments, always use HTTPS:

#### Nginx Configuration Example
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 Monitoring & Logging

### Health Checks
Most platforms support health checks. The app includes a health endpoint that can be monitored.

### Logging
Add logging to your Streamlit app:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In your app
logger.info("Model training started")
```

### Performance Monitoring
Consider adding:
- Application Performance Monitoring (APM) tools
- Error tracking (e.g., Sentry)
- Usage analytics

---

## 🛠️ Troubleshooting

### Common Issues

1. **Port Issues**
   - Ensure the correct port is exposed and configured
   - Use `--server.port $PORT` for Heroku/Railway

2. **Memory Issues**
   - Large datasets may cause memory problems
   - Consider using data streaming or pagination

3. **Package Compatibility**
   - Pin specific versions in requirements.txt
   - Test locally before deploying

4. **File Upload Issues**
   - Check file size limits on your platform
   - Implement proper error handling

### Debug Mode
Enable debug mode for development:
```bash
streamlit run streamlit_app.py --server.runOnSave true --server.port 8501
```

---

## 📈 Scaling Considerations

### For High Traffic
- Use load balancers
- Implement caching mechanisms
- Consider container orchestration (Kubernetes)
- Database integration for model persistence

### Performance Optimization
- Use `@st.cache` for expensive computations
- Implement lazy loading for large datasets
- Optimize model loading and prediction times

---

## 🔄 CI/CD Pipeline

The included GitHub Actions workflow provides:
- Automated testing on push/PR
- Dependency checks
- Syntax validation

Enhance with:
- Automated deployment
- Security scanning
- Performance testing

---

## 📞 Support

If you encounter issues during deployment:
1. Check the platform-specific documentation
2. Review logs for error messages
3. Verify all dependencies are correctly installed
4. Test locally first

For platform-specific help:
- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io/streamlit-cloud)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **Docker**: [docs.docker.com](https://docs.docker.com)

---

## ✅ Deployment Checklist

Before deploying:
- [ ] Test application locally
- [ ] Update requirements.txt
- [ ] Set up environment variables
- [ ] Configure proper error handling
- [ ] Test with sample data
- [ ] Setup monitoring/logging
- [ ] Configure HTTPS (production)
- [ ] Document deployment process
- [ ] Create backup strategy

Your BRCA Cancer Prediction System is now ready for deployment! 🚀
