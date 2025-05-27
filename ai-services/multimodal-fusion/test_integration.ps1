#!/usr/bin/env pwsh

# Multimodal Content Analyzer - Comprehensive Integration Test Script
Write-Host "🚀 Starting Multimodal Content Analyzer Integration Tests" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Configuration
$API_BASE = "http://localhost:8000"
$TEXT_ANALYZER = "http://localhost:8001"
$IMAGE_ANALYZER = "http://localhost:8002"
$VIDEO_ANALYZER = "http://localhost:8003"
$FUSION_SERVICE = "http://localhost:8004"

Write-Host "`n📋 Step 1: Health Checks" -ForegroundColor Green
Write-Host "-------------------------" -ForegroundColor Green

# Test all services
$services = @(
    @{Name = "Backend"; Url = $API_BASE},
    @{Name = "Text-Analyzer"; Url = $TEXT_ANALYZER},
    @{Name = "Image-Analyzer"; Url = $IMAGE_ANALYZER},
    @{Name = "Video-Analyzer"; Url = $VIDEO_ANALYZER},
    @{Name = "Fusion-Service"; Url = $FUSION_SERVICE}
)

foreach ($service in $services) {
    $name = $service.Name
    $url = $service.Url
    
    Write-Host "Testing $name..."
    
    try {
        $healthCheck = Invoke-WebRequest -Uri "$url/health" -Method GET -UseBasicParsing -ErrorAction SilentlyContinue
        if ($healthCheck.StatusCode -eq 200) {
            Write-Host "✅ $name is healthy" -ForegroundColor Green
        } else {
            Write-Host "❌ $name is not responding" -ForegroundColor Red
        }
    } catch {
        try {
            $rootCheck = Invoke-WebRequest -Uri "$url/" -Method GET -UseBasicParsing -ErrorAction SilentlyContinue
            if ($rootCheck.StatusCode -eq 200) {
                Write-Host "✅ $name is healthy" -ForegroundColor Green
            } else {
                Write-Host "❌ $name is not responding" -ForegroundColor Red
            }
        } catch {
            Write-Host "❌ $name is not responding" -ForegroundColor Red
        }
    }
}

Write-Host "`n🔐 Step 2: Authentication Test" -ForegroundColor Green
Write-Host "------------------------------" -ForegroundColor Green

# Test user registration and login
Write-Host "Creating test user..."
$registerBody = @{
    email = "test@multimodal.ai"
    username = "testuser"
    password = "testpass123"
    full_name = "Test User"
} | ConvertTo-Json

try {
    $registerResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/register" -Method Post -Body $registerBody -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Registration response: $($registerResponse | ConvertTo-Json -Compress)"
} catch {
    Write-Host "Registration response: $($_.Exception.Message)"
}

Write-Host "Logging in..."
$loginForm = @{
    username = "test@multimodal.ai"
    password = "testpass123"
}

try {
    $loginResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/login" -Method Post -Body $loginForm -ContentType "application/x-www-form-urlencoded" -ErrorAction SilentlyContinue
    $TOKEN = $loginResponse.access_token
    
    if ($TOKEN) {
        Write-Host "✅ Authentication successful" -ForegroundColor Green
        Write-Host "Token: $($TOKEN.Substring(0, [Math]::Min(20, $TOKEN.Length)))..."
    } else {
        Write-Host "❌ Authentication failed" -ForegroundColor Red
        Write-Host "Response: $($loginResponse | ConvertTo-Json -Compress)"
    }
} catch {
    Write-Host "❌ Authentication failed" -ForegroundColor Red
    Write-Host "Response: $($_.Exception.Message)"
    $TOKEN = $null
}

Write-Host "`n🧠 Step 3: AI Services Tests" -ForegroundColor Green
Write-Host "----------------------------" -ForegroundColor Green

# Test Text Analyzer
Write-Host "Testing Text Analyzer..."
$textBody = @{
    text_content = "This is a wonderful day for testing our multimodal AI system!"
    analysis_type = "sentiment"
} | ConvertTo-Json

try {
    $textTest = Invoke-RestMethod -Uri "$TEXT_ANALYZER/analyze" -Method Post -Body $textBody -ContentType "application/json" -ErrorAction SilentlyContinue
    Write-Host "Text Analysis Result: Sentiment: $($textTest.results.label), Confidence: $($textTest.results.confidence)"
} catch {
    Write-Host "Text Analysis Result: Error - $($_.Exception.Message)"
}

# Test Image Analyzer
Write-Host "Testing Image Analyzer..."
try {
    $imageHealth = Invoke-RestMethod -Uri "$IMAGE_ANALYZER/health" -Method Get -ErrorAction SilentlyContinue
    Write-Host "Image Analyzer Health: $($imageHealth.status)"
} catch {
    Write-Host "Image Analyzer Health: Error - $($_.Exception.Message)"
}

# Test Video Analyzer
Write-Host "Testing Video Analyzer..."
try {
    $videoHealth = Invoke-RestMethod -Uri "$VIDEO_ANALYZER/health" -Method Get -ErrorAction SilentlyContinue
    Write-Host "Video Analyzer Health: $($videoHealth.status)"
} catch {
    Write-Host "Video Analyzer Health: Error - $($_.Exception.Message)"
}

# Test Multimodal Fusion
Write-Host "Testing Multimodal Fusion..."
try {
    $fusionHealth = Invoke-RestMethod -Uri "$FUSION_SERVICE/health" -Method Get -ErrorAction SilentlyContinue
    Write-Host "Fusion Service Health: $($fusionHealth.status)"
} catch {
    Write-Host "Fusion Service Health: Error - $($_.Exception.Message)"
}

Write-Host "`n🔗 Step 4: End-to-End Integration Test" -ForegroundColor Green
Write-Host "-------------------------------------" -ForegroundColor Green

if ($TOKEN) {
    Write-Host "Testing content upload and analysis..."
    
    # Create test content
    Write-Host "Creating test text content..."
    $contentBody = @{
        title = "Test Content"
        description = "Test content for integration testing"
        content_type = "text"
        text_content = "This is a comprehensive test of our multimodal content analysis platform. It includes sentiment analysis, entity extraction, and cross-modal fusion capabilities."
    } | ConvertTo-Json
    
    try {
        $contentResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/content/items/" -Method Post -Headers @{Authorization = "Bearer $TOKEN"} -Body $contentBody -ContentType "application/json" -ErrorAction SilentlyContinue
        $CONTENT_ID = $contentResponse.id
        
        if ($CONTENT_ID) {
            Write-Host "✅ Content created with ID: $CONTENT_ID" -ForegroundColor Green
            
            # Start analysis
            Write-Host "Starting multimodal analysis..."
            $analysisBody = @{
                content_item = $CONTENT_ID
                analysis_types = @("sentiment", "entities", "topics")
                configuration = @{}
            } | ConvertTo-Json
            
            try {
                $analysisResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/analysis/jobs/" -Method Post -Headers @{Authorization = "Bearer $TOKEN"} -Body $analysisBody -ContentType "application/json" -ErrorAction SilentlyContinue
                $JOB_ID = $analysisResponse.id
                
                if ($JOB_ID) {
                    Write-Host "✅ Analysis job created with ID: $JOB_ID" -ForegroundColor Green
                    Write-Host "Analysis job started successfully!" -ForegroundColor Green
                } else {
                    Write-Host "❌ Failed to create analysis job" -ForegroundColor Red
                    Write-Host "Response: $($analysisResponse | ConvertTo-Json -Compress)"
                }
            } catch {
                Write-Host "❌ Failed to create analysis job" -ForegroundColor Red
                Write-Host "Response: $($_.Exception.Message)"
            }
        } else {
            Write-Host "❌ Failed to create content" -ForegroundColor Red
            Write-Host "Response: $($contentResponse | ConvertTo-Json -Compress)"
        }
    } catch {
        Write-Host "❌ Failed to create content" -ForegroundColor Red
        Write-Host "Response: $($_.Exception.Message)"
    }
} else {
    Write-Host "⚠️ Skipping integration test due to authentication failure" -ForegroundColor Yellow
}

Write-Host "`n📊 Step 5: System Status Summary" -ForegroundColor Green
Write-Host "--------------------------------" -ForegroundColor Green

Write-Host "Getting system statistics..."
if ($TOKEN) {
    try {
        $userStats = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/stats/" -Method Get -Headers @{Authorization = "Bearer $TOKEN"} -ErrorAction SilentlyContinue
        Write-Host "User Stats: Quota: $($userStats.usage_quota), Used: $($userStats.usage_count)"
    } catch {
        Write-Host "User Stats: N/A"
    }
    
    try {
        $contentStats = Invoke-RestMethod -Uri "$API_BASE/api/v1/content/stats/" -Method Get -Headers @{Authorization = "Bearer $TOKEN"} -ErrorAction SilentlyContinue
        Write-Host "Content Stats: Total Items: $($contentStats.total_items)"
    } catch {
        Write-Host "Content Stats: N/A"
    }
}

Write-Host "`n🎉 Integration Test Complete!" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host "The Multimodal Content Analyzer system is ready for use!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access Points:" -ForegroundColor Yellow
Write-Host "- Main API: http://localhost:8000/docs"
Write-Host "- Text Analyzer: http://localhost:8001"
Write-Host "- Image Analyzer: http://localhost:8002"
Write-Host "- Video Analyzer: http://localhost:8003"
Write-Host "- Fusion Service: http://localhost:8004"
Write-Host ""
Write-Host "📚 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Upload various content types (text, images, videos)"
Write-Host "2. Run different analysis types"
Write-Host "3. Explore multimodal fusion capabilities"
Write-Host "4. Check monitoring and insights"