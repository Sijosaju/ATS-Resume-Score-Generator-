document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const jobDescription = document.getElementById('jobDescription');
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    const improvementTipsBtn = document.getElementById('improvementTipsBtn');
    const tipsModal = document.getElementById('tipsModal');
    const tipsContent = document.getElementById('tipsContent');
    const closeBtn = document.querySelector('#tipsModal .close-btn');
  
    // Global variable to store analysis result
    let analysisResult = null;
  
    // Handle upload area click
    uploadArea.addEventListener('click', function() {
      fileInput.click();
    });
  
    // Handle file selection
    fileInput.addEventListener('change', function() {
      if (this.files && this.files[0]) {
        const fileName = this.files[0].name;
        uploadArea.querySelector('h3').textContent = `File Selected: ${fileName}`;
        uploadArea.querySelector('p').textContent = 'Click "Analyze Resume" to continue';
      }
    });
  
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      this.style.backgroundColor = 'rgba(67, 97, 238, 0.1)';
      this.style.borderColor = '#4361ee';
    });
  
    uploadArea.addEventListener('dragleave', function() {
      this.style.backgroundColor = '';
      this.style.borderColor = '#ccc';
    });
  
    uploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      this.style.backgroundColor = '';
      this.style.borderColor = '#ccc';
  
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        const fileName = e.dataTransfer.files[0].name;
        this.querySelector('h3').textContent = `File Selected: ${fileName}`;
        this.querySelector('p').textContent = 'Click "Analyze Resume" to continue';
      }
    });
  
    // Show loading indicator
    function showLoading() {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Analyzing...';
    }
  
    // Hide loading indicator
    function hideLoading() {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze Resume';
    }
  
    // Update results with data from backend
    function updateResults(data) {
      // Store the analysis result for later use (download & tips)
      analysisResult = data;
  
      // Update score circle
      const scoreValue = document.querySelector('.score-value');
      scoreValue.textContent = `${data.overall_score}%`;
  
      // Update score circle gradient
      const scoreCircle = document.querySelector('.score-circle');
      scoreCircle.style.background = `conic-gradient(var(--primary) 0% ${data.overall_score}%, #f0f0f0 ${data.overall_score}% 100%)`;
  
      // Update feedback
      const feedbackList = document.querySelector('.feedback-list');
      feedbackList.innerHTML = '<h3>Feedback & Suggestions</h3>';
  
      if (data.feedback && Array.isArray(data.feedback)) {
        data.feedback.forEach(item => {
          const iconClass = item.type === 'success' ? 'fa-check-circle icon-check' : 
                            item.type === 'warning' ? 'fa-exclamation-triangle icon-warning' : 
                            item.type === 'danger' ? 'fa-times-circle icon-danger' : 'fa-info-circle icon-info';
  
          const feedbackItem = document.createElement('div');
          feedbackItem.className = `feedback-item ${item.type}`;
          feedbackItem.innerHTML = `
            <h4><i class="fas ${iconClass} icon"></i> ${item.title}</h4>
            <p>${item.description}</p>
          `;
          feedbackList.appendChild(feedbackItem);
        });
      }
  
      // Update skill matches
      const skillMatch = document.querySelector('.skill-match');
      skillMatch.innerHTML = '<h3>Key Skills Match</h3>';
  
      if (data.keyword_matches && Object.keys(data.keyword_matches).length > 0) {
        const skillMatches = Object.entries(data.keyword_matches).map(([skill, matchType]) => {
          return {
            skill: skill,
            match_percentage: matchType === "exact match" ? 100 : 50
          };
        }).slice(0, 6);
  
        skillMatches.forEach(skill => {
          const skillItem = document.createElement('div');
          skillItem.className = 'skill-item';
          skillItem.innerHTML = `
            <div class="skill-label">
              <span>${skill.skill}</span>
              <span class="match-percentage">${skill.match_percentage}%</span>
            </div>
            <div class="skill-bar">
              <div class="skill-progress" style="width: ${skill.match_percentage}%"></div>
            </div>
          `;
          skillMatch.appendChild(skillItem);
        });
      } else if (data.extracted_job_keywords && data.extracted_job_keywords.length > 0) {
        data.extracted_job_keywords.slice(0, 6).forEach(keyword => {
          const skillItem = document.createElement('div');
          skillItem.className = 'skill-item';
          skillItem.innerHTML = `
            <div class="skill-label">
              <span>${keyword}</span>
              <span class="match-percentage">0%</span>
            </div>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 0%"></div>
            </div>
          `;
          skillMatch.appendChild(skillItem);
        });
      } else {
        skillMatch.innerHTML += '<p>No skill match data available</p>';
      }
  
      // Add overall assessment text
      const scoreCard = document.querySelector('.score-card');
      let assessmentText;
  
      if (data.overall_score >= 80) {
        assessmentText = `<h3>Your resume is highly ATS-friendly!</h3>
                          <p>You have a strong match with this job description and good ATS compatibility.</p>`;
      } else if (data.overall_score >= 60) {
        assessmentText = `<h3>Your resume is moderately ATS-friendly</h3>
                          <p>Your resume has good ATS compatibility but could be improved for this specific job.</p>`;
      } else {
        assessmentText = `<h3>Your resume needs ATS optimization</h3>
                          <p>To improve your chances, consider updating your resume based on the feedback below.</p>`;
      }
  
      scoreCard.innerHTML = `
        <div class="score-circle">
          <div class="score-value">${data.overall_score}%</div>
        </div>
        ${assessmentText}
      `;
  
      // Display results section
      resultsSection.style.display = 'block';
      resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
  
    // Handle form submission
    analyzeBtn.addEventListener('click', function() {
      if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select a resume file first.');
        return;
      }
  
      showLoading();
  
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('job_description', jobDescription.value);
  
      fetch('/analyze', {
        method: 'POST',
        body: formData
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          hideLoading();
          updateResults(data);
        })
        .catch(error => {
          hideLoading();
          alert('Error analyzing resume: ' + error.message);
        });
    });
  
    // Download Report functionality
    downloadReportBtn.addEventListener('click', function() {
      if (!analysisResult) {
        alert('No analysis data available. Please analyze your resume first.');
        return;
      }
      // Generate a simple text report
      let report = `ATS Resume Analysis Report\n\nOverall Score: ${analysisResult.overall_score}%\n\n---\nFeedback & Suggestions:\n`;
      if (analysisResult.feedback && analysisResult.feedback.length) {
        analysisResult.feedback.forEach(item => {
          report += `• ${item.title}: ${item.description}\n`;
        });
      }
      report += `\n---\nKey Skills Match:\n`;
      if (analysisResult.keyword_matches) {
        for (const [skill, matchType] of Object.entries(analysisResult.keyword_matches)) {
          report += `• ${skill}: ${matchType}\n`;
        }
      }
      // You can append more details from analysisResult as needed
  
      // Create a blob and trigger download
      const blob = new Blob([report], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ATS_Resume_Report.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  
    // Get Improvement Tips functionality
    improvementTipsBtn.addEventListener('click', function() {
      if (!analysisResult) {
        alert('No analysis data available. Please analyze your resume first.');
        return;
      }
      // Gather improvement tips from both formatting tips and relevant feedback (warning/danger)
      let tips = [];
      if (analysisResult.formatting_tips && analysisResult.formatting_tips.length) {
        tips.push("Formatting Tips: " + analysisResult.formatting_tips.join('; '));
      }
      // Include any feedback that suggests improvements
      if (analysisResult.feedback && analysisResult.feedback.length) {
        analysisResult.feedback.forEach(item => {
          if (item.type === 'warning' || item.type === 'danger') {
            tips.push(`${item.title}: ${item.description}`);
          }
        });
      }
      if (tips.length === 0) {
        tips.push("No specific improvement tips available. Your resume seems well-optimized!");
      }
      // Display tips in modal
      tipsContent.innerHTML = '<ul>' + tips.map(tip => `<li>${tip}</li>`).join('') + '</ul>';
      tipsModal.style.display = 'flex';
    });
  
    // Close the modal when clicking the close button
    closeBtn.addEventListener('click', function() {
      tipsModal.style.display = 'none';
    });
  
    // Also close modal if clicking outside the modal content
    window.addEventListener('click', function(e) {
      if (e.target === tipsModal) {
        tipsModal.style.display = 'none';
      }
    });
  });
  