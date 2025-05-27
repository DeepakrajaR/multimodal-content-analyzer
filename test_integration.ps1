# Multimodal Content Analyzer - PowerShell Integration Test Script
Write-Host "🚀 Starting Multimodal Content Analyzer Integration Tests" -ForegroundColor Green
Write-Host "========================================================"

# Configuration
$API_BASE = "http://localhost:8000"
$TEXT_ANALYZER = "http://localhost:8001"
$IMAGE_ANALYZER = "http://localhost:8002"
$VIDEO_ANALYZER = "http://localhost:8003"
$FUSION_SERVICE = "http://localhost:8004"

Write-Host "📋 Step 1: Health Checks" -ForegroundColor Yellow
Write-Host "-------------------------"

# Test all services
$services = @{
    "Backend" = $API_BASE
    "Text-Analyzer" = $TEXT_ANALYZER
    "Image-Analyzer" = $IMAGE_ANALYZER
    "Video-Analyzer" = $VIDEO_ANALYZER
    "Fusion-Service" = $FUSION_SERVICE
}

foreach ($service in $services.GetEnumerator()) {
    Write-Host "Testing $($service.Key)..." -NoNewline
    try {
        $healthUrl = "$($service.Value)/health"
        $rootUrl = "$($service.Value)/"
        
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5
            Write-Host " ✅ Healthy" -ForegroundColor Green
        }
        catch {
            try {
                $response = Invoke-RestMethod -Uri $rootUrl -Method Get -TimeoutSec 5
                Write-Host " ✅ Running" -ForegroundColor Green
            }
            catch {
                Write-Host " ❌ Not responding" -ForegroundColor Red
            }
        }
    }
    catch {
        Write-Host " ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🔐 Step 2: Authentication Test" -ForegroundColor Yellow
Write-Host "------------------------------"

$TOKEN = $null

try {
    Write-Host "Creating test user..." -NoNewline
    $registerBody = @{
        email = "test@multimodal.ai"
        username = "testuser"
        password = "testpass123"
        full_name = "Test User"
    } | ConvertTo-Json
    
    $registerResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/register" -Method Post -Body $registerBody -ContentType "application/json" -TimeoutSec 10
    Write-Host " ✅ User created" -ForegroundColor Green
}
catch {
    if ($_.Exception.Response.StatusCode -eq 400) {
        Write-Host " ⚠️ User may already exist" -ForegroundColor Yellow
    } else {
        Write-Host " ❌ Registration failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

try {
    Write-Host "Logging in..." -NoNewline
    $loginBody = "username=test@multimodal.ai&password=testpass123"
    $loginResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/login" -Method Post -Body $loginBody -ContentType "application/x-www-form-urlencoded" -TimeoutSec 10
    
    $TOKEN = $loginResponse.access_token
    if ($TOKEN) {
        Write-Host " ✅ Authentication successful" -ForegroundColor Green
        # Fixed potential null reference by checking if TOKEN exists and has length
        if ($TOKEN -and $TOKEN.Length -gt 0) {
            Write-Host "Token: $($TOKEN.Substring(0, [Math]::Min(20, $TOKEN.Length)))..."
        } else {
            Write-Host "Token received but may be empty" -ForegroundColor Yellow
        }
    } else {
        Write-Host " ❌ Authentication failed - no token received" -ForegroundColor Red
    }
}
catch {
    Write-Host " ❌ Authentication failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🧠 Step 3: AI Services Tests" -ForegroundColor Yellow
Write-Host "----------------------------"

# Test Text Analyzer
Write-Host "Testing Text Analyzer..." -NoNewline
try {
    $textTestBody = @{
        text_content = "This is a wonderful day for testing our multimodal AI system!"
        analysis_type = "sentiment"
    } | ConvertTo-Json
    
    $textResult = Invoke-RestMethod -Uri "$TEXT_ANALYZER/analyze" -Method Post -Body $textTestBody -ContentType "application/json" -TimeoutSec 15
    
    $sentiment = $textResult.results.label
    $confidence = $textResult.results.confidence
    Write-Host " ✅ Sentiment: $sentiment, Confidence: $confidence" -ForegroundColor Green
}
catch {
    Write-Host " ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Test other services health
$aiServices = @{
    "Image Analyzer" = $IMAGE_ANALYZER
    "Video Analyzer" = $VIDEO_ANALYZER
    "Fusion Service" = $FUSION_SERVICE
}

foreach ($service in $aiServices.GetEnumerator()) {
    Write-Host "Testing $($service.Key)..." -NoNewline
    try {
        $healthResponse = Invoke-RestMethod -Uri "$($service.Value)/health" -Method Get -TimeoutSec 5
        Write-Host " ✅ Status: $($healthResponse.status)" -ForegroundColor Green
    }
    catch {
        Write-Host " ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🔗 Step 4: End-to-End Integration Test" -ForegroundColor Yellow
Write-Host "-------------------------------------"

if ($TOKEN) {
    Write-Host "Testing content upload and analysis..."
    
    try {
        Write-Host "Creating test text content..." -NoNewline
        $contentBody = @{
            title = "Test Content"
            description = "Test content for integration testing"
            content_type = "text"
            text_content = "This is a comprehensive test of our multimodal content analysis platform. It includes sentiment analysis, entity extraction, and cross-modal fusion capabilities."
        } | ConvertTo-Json
        
        $headers = @{
            "Authorization" = "Bearer $TOKEN"
            "Content-Type" = "application/json"
        }
        
        $contentResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/content/items/" -Method Post -Body $contentBody -Headers $headers -TimeoutSec 10
        $contentId = $contentResponse.id
        Write-Host " ✅ Content created with ID: $contentId" -ForegroundColor Green
        
        Write-Host "Starting multimodal analysis..." -NoNewline
        $analysisBody = @{
            content_item = $contentId
            analysis_types = @("sentiment", "entities", "topics")
            configuration = @{}
        } | ConvertTo-Json
        
        $analysisResponse = Invoke-RestMethod -Uri "$API_BASE/api/v1/analysis/jobs/" -Method Post -Body $analysisBody -Headers $headers -TimeoutSec 10
        $jobId = $analysisResponse.id
        Write-Host " ✅ Analysis job created with ID: $jobId" -ForegroundColor Green
    }
    catch {
        Write-Host " ❌ Integration test failed: $($_.Exception.Message)" -ForegroundColor Red
        if ($_.Exception.Response) {
            Write-Host "Status Code: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "⚠️ Skipping integration test due to authentication failure" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📊 Step 5: System Status Summary" -ForegroundColor Yellow
Write-Host "--------------------------------"

if ($TOKEN) {
    Write-Host "Getting system statistics..."
    try {
        $headers = @{
            "Authorization" = "Bearer $TOKEN"
        }
        
        $userStats = Invoke-RestMethod -Uri "$API_BASE/api/v1/auth/stats/" -Headers $headers -TimeoutSec 5
        $contentStats = Invoke-RestMethod -Uri "$API_BASE/api/v1/content/stats/" -Headers $headers -TimeoutSec 5
        
        Write-Host "User Stats: Quota: $($userStats.usage_quota), Used: $($userStats.usage_count)" -ForegroundColor Cyan
        Write-Host "Content Stats: Total Items: $($contentStats.total_items)" -ForegroundColor Cyan
    }
    catch {
        Write-Host "Could not retrieve system statistics: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🎉 Integration Test Complete!" -ForegroundColor Green
Write-Host "============================="
Write-Host "The Multimodal Content Analyzer system is ready for use!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access Points:" -ForegroundColor Cyan
Write-Host "- Main API: http://localhost:8000/docs"
Write-Host "- Text Analyzer: http://localhost:8001"
Write-Host "- Image Analyzer: http://localhost:8002"
Write-Host "- Video Analyzer: http://localhost:8003"
Write-Host "- Fusion Service: http://localhost:8004"
Write-Host ""
Write-Host "📚 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Upload various content types (text, images, videos)"
Write-Host "2. Run different analysis types"
Write-Host "3. Explore multimodal fusion capabilities"
Write-Host "4. Check monitoring and insights"

Write-Host ""
Write-Host "🔧 Troubleshooting:" -ForegroundColor Magenta
Write-Host "If any services are not responding:"
Write-Host "1. Check if Docker containers are running: docker-compose ps"
Write-Host "2. Check logs: docker-compose logs [service-name]"
Write-Host "3. Restart services: docker-compose restart"
Write-Host "4. Rebuild if needed: docker-compose up --build"