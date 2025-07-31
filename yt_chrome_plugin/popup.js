const API_BASE_URL = "http://127.0.0.1:5000"; // Replace with your backend IP if needed
const API_KEY = "Youtube api key ";

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const url = document.getElementById("videoUrl").value;
  const videoId = extractVideoId(url);
  const outputDiv = document.getElementById("output");
  if (!videoId) {
    outputDiv.innerHTML = "Invalid YouTube URL.";
    return;
  }

  outputDiv.innerHTML = "Fetching comments...";
  const comments = await fetchComments(videoId);
  if (comments.length === 0) {
    outputDiv.innerHTML = "No comments found.";
    return;
  }

  outputDiv.innerHTML = "Analyzing sentiment...";
  try {
    const response = await fetch(`${API_BASE_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments })
    });

    const result = await response.json();
    if (!Array.isArray(result)) {
      outputDiv.innerHTML = `Error: ${result.error || "Unexpected error."}`;
      return;
    }

    displayResults(result);
    await generateChart(result);
    await generateWordCloud(comments);
    await generateTrendGraph(result);
  } catch (error) {
    console.error("Error:", error);
    outputDiv.innerHTML = "Error fetching sentiment predictions.";
  }
});

function extractVideoId(url) {
  const regex = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w\-]+)/;
  const match = url.match(regex);
  return match ? match[1] : null;
}

async function fetchComments(videoId) {
  const comments = [];
  let pageToken = "";
  try {
    while (comments.length < 300) {
      const res = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
      const data = await res.json();

      if (data.items) {
        for (const item of data.items) {
          const snippet = item.snippet.topLevelComment.snippet;
          comments.push({
            text: snippet.textOriginal,
            timestamp: snippet.publishedAt,
            authorId: snippet.authorChannelId?.value || "Unknown"
          });
        }
      }

      pageToken = data.nextPageToken;
      if (!pageToken) break;
    }
  } catch (e) {
    console.error("Comment fetch error:", e);
    document.getElementById("output").innerHTML = "Error fetching comments.";
  }

  return comments;
}

function displayResults(data) {
  const outputDiv = document.getElementById("output");
  outputDiv.innerHTML = "<h3>Sentiment Analysis Results:</h3>";
  data.forEach(item => {
    const sentiment = item.sentiment === 1 ? "Positive" : item.sentiment === 0 ? "Neutral" : "Negative";
    outputDiv.innerHTML += `
      <div style="margin-bottom:10px;">
        <strong>Comment:</strong> ${item.comment}<br/>
        <strong>Sentiment:</strong> ${sentiment}<br/>
        <strong>Timestamp:</strong> ${item.timestamp}
      </div>
    `;
  });
}

async function generateChart(data) {
  const sentimentCounts = { "-1": 0, "0": 0, "1": 0 };
  for (const item of data) {
    sentimentCounts[item.sentiment] = (sentimentCounts[item.sentiment] || 0) + 1;
  }

  try {
    const res = await fetch(`${API_BASE_URL}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: sentimentCounts })
    });

    const blob = await res.blob();
    document.getElementById("pieChart").src = URL.createObjectURL(blob);
  } catch (err) {
    console.error("Error loading pie chart:", err);
  }
}

async function generateWordCloud(comments) {
  try {
    const texts = comments.map(c => c.text);
    const res = await fetch(`${API_BASE_URL}/generate_wordcloud`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: texts })
    });

    const blob = await res.blob();
    document.getElementById("wordCloud").src = URL.createObjectURL(blob);
  } catch (err) {
    console.error("Error loading word cloud:", err);
  }
}

async function generateTrendGraph(data) {
  try {
    const res = await fetch(`${API_BASE_URL}/generate_trend_graph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_data: data })
    });

    const blob = await res.blob();
    document.getElementById("trendGraph").src = URL.createObjectURL(blob);
  } catch (err) {
    console.error("Error loading trend graph:", err);
  }
}
