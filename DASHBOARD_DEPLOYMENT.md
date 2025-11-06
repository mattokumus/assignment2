# ECHR Case Explorer - Dashboard Deployment Guide

**Interactive Dashboard for ECHR Case Analysis**

This guide explains how to run and deploy the Streamlit dashboard for exploring the ECHR case dataset.

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.8 or higher
- Dependencies installed from `requirements.txt`

### Running Locally

```bash
# 1. Ensure all dependencies are installed
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Open browser (should auto-open)
# Default URL: http://localhost:8501
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🌐 Deploy to Streamlit Cloud (FREE!)

Streamlit Cloud offers free hosting for public repositories.

### Step 1: Push to GitHub

```bash
# Ensure all files are committed
git add app.py requirements.txt DASHBOARD_DEPLOYMENT.md
git commit -m "Add interactive Streamlit dashboard"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Visit:** https://share.streamlit.io/

2. **Sign in** with GitHub account

3. **Create new app:**
   - Repository: `mattokumus/assignment2`
   - Branch: `main` (or your current branch)
   - Main file path: `app.py`

4. **Click "Deploy"**

5. **Wait 2-3 minutes** for deployment

6. **Get shareable URL:**
   - Format: `https://[your-app-name].streamlit.app`
   - Example: `https://echr-case-explorer.streamlit.app`

### Step 3: Share Your Dashboard

Once deployed, you can share the URL with:
- ✅ Researchers and colleagues
- ✅ In your thesis/dissertation
- ✅ On social media
- ✅ In academic presentations

**No server maintenance required!** Streamlit Cloud handles everything.

---

## 📊 Dashboard Features

### Sidebar Filters

| Filter | Type | Options |
|--------|------|---------|
| **Countries** | Multiselect | All 45 countries (default: top 10) |
| **Region** | Radio | All / Eastern Europe / Western Europe |
| **Year Range** | Slider | 2000-2024 |
| **Articles** | Multiselect | All articles (default: top 5) |
| **Violation Status** | Radio | All / Violations Only / No Violations |
| **Applicant Type** | Multiselect | Individual, NGO, Government, etc. |

### Key Statistics

Dashboard displays 5 real-time metrics:
- **Total Cases** (filtered count + % of dataset)
- **Violation Rate** (% with delta vs overall)
- **Countries** (unique count)
- **Articles** (unique count)
- **Time Span** (date range)

### Visualization Tabs

#### 📍 Country Analysis
- **Violation Rate by Country** (bar chart)
- **Case Count Distribution** (treemap)

#### 📜 Article Analysis
- **Article Distribution** (pie chart)
- **Violation Rate by Article** (bar chart)

#### 🗺️ Regional Comparison
- **Regional Violation Rates** (bar chart with comparison)
- **Regional Case Distribution** (pie chart)

#### 📈 Temporal Trends
- **Cases Over Time** (area chart)
- **Violation Rate Trends** (line chart)
- **Regional Trends** (line chart comparing East vs West)

### Data Table

- **Display:** First 100 rows (for performance)
- **Features:**
  - Sortable columns
  - Searchable (use browser Ctrl+F)
  - Formatted columns (checkboxes for violations)
- **Download:** Full filtered dataset as CSV (with timestamp)

---

## ⚙️ Configuration Options

### Customizing the Dashboard

Edit `app.py` to customize:

**1. Page Title and Icon**
```python
st.set_page_config(
    page_title="ECHR Case Explorer",  # Browser tab title
    page_icon="⚖️",                    # Favicon
    layout="wide"                       # Layout mode
)
```

**2. Default Filters**
```python
# Change default countries (currently top 10)
default_countries = top_countries[:10]  # Modify number

# Change default articles (currently top 5)
default_articles = top_articles[:5]     # Modify number
```

**3. Color Scheme**
```python
# Regional colors (in visualizations)
color_discrete_map = {
    'Eastern Europe': '#ff7f0e',  # Orange
    'Western Europe': '#1f77b4'   # Blue
}
```

**4. Data Table Display**
```python
# Show more/fewer rows
st.dataframe(display_df[display_columns].head(100))  # Change 100
```

### Performance Tuning

**For large datasets:**

```python
# Increase cache TTL (time to live)
@st.cache_data(ttl=3600)  # 1 hour
def load_data():
    ...

# Limit visualizations
max_countries_to_show = 20  # Add in plotting code
```

**For faster loading:**
- Pre-filter data in `load_data()` function
- Reduce default selections
- Use `st.spinner()` for loading indicators

---

## 🐛 Troubleshooting

### Dashboard won't start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip install streamlit plotly
# or
pip install -r requirements.txt
```

---

**Error:** `FileNotFoundError: extracted_data.csv`

**Solution:**
```bash
# Generate the CSV first
python3 assignment2.py

# Then run dashboard
streamlit run app.py
```

---

### Filters not working

**Issue:** Selections don't update visualizations

**Solution:**
- Hard refresh browser: `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)
- Clear Streamlit cache: Click "⋮" menu → "Clear cache"
- Restart dashboard: `Ctrl + C` in terminal, then `streamlit run app.py` again

---

### Visualizations too slow

**Issue:** Charts take long to render

**Solutions:**
1. **Reduce data points:**
   ```python
   # In app.py, add sampling
   if len(filtered_df) > 5000:
       sampled_df = filtered_df.sample(5000)
   ```

2. **Limit default selections:**
   - Set default countries to top 5 instead of 10
   - Use fewer default articles

3. **Use aggregated data:**
   - Pre-aggregate in `load_data()` function
   - Display summaries instead of raw data

---

### Deployment fails on Streamlit Cloud

**Error:** "requirements.txt not found"

**Solution:** Ensure `requirements.txt` is in repository root (not subdirectory)

---

**Error:** "App crashes on startup"

**Solution:**
1. Check Streamlit Cloud logs (click "Manage app" → "Logs")
2. Common issues:
   - Missing `extracted_data.csv` → Add to Git LFS or generate in app
   - Incompatible package versions → Pin versions in requirements.txt
   - Memory limits → Reduce data size or use Streamlit's paid tier

---

## 📱 Mobile Support

The dashboard is responsive but works best on desktop/tablet. For mobile users:

**Recommendations:**
- Use landscape orientation
- Collapse sidebar after selecting filters (click X)
- Use "Fullscreen" mode for visualizations (hover → expand icon)

---

## 🔒 Privacy and Security

**Public Deployment:**
- ✅ No personal data exposed (all ECHR cases are public record)
- ✅ No authentication required (read-only dashboard)
- ✅ No user data collection

**Private Deployment:**
If you need private hosting:
1. Deploy on Streamlit for Teams (paid)
2. Self-host with Docker (see below)
3. Use Heroku/AWS with authentication

---

## 🐳 Docker Deployment (Advanced)

For self-hosting with Docker:

**1. Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**2. Build and run:**
```bash
docker build -t echr-dashboard .
docker run -p 8501:8501 echr-dashboard
```

**3. Access:**
Open `http://localhost:8501`

---

## 📊 Usage Tips

### For Researchers

1. **Export filtered data:**
   - Apply filters → Click "Download Filtered Data as CSV"
   - Use in your own analysis (R, Python, STATA, etc.)

2. **Create screenshots:**
   - Use browser screenshot tools
   - Include in presentations/papers
   - Cite: "Interactive dashboard available at [URL]"

3. **Share specific views:**
   - URL parameters don't persist (Streamlit limitation)
   - Instead: Document filter settings in papers
   - Example: "Dashboard filtered to: Eastern Europe, 2010-2020, Article 3"

### For Students

1. **Explore patterns:**
   - Try different country combinations
   - Compare violation rates across regions
   - Identify temporal trends

2. **Generate hypotheses:**
   - "Why does Country X have higher violation rate?"
   - "Did violations decrease after year Y?"
   - Test with statistical analysis scripts

3. **Learn interactively:**
   - See how filters affect results in real-time
   - Understand data structure from table view
   - Visualize abstract concepts (violation rates, trends)

---

## 🔄 Updating the Dashboard

### Adding New Features

**1. Add new visualization:**
```python
# In app.py, add new tab or section
with tab5:  # Add fifth tab
    st.subheader("📊 My New Analysis")

    fig = px.scatter(
        filtered_df,
        x='year',
        y='violation_count',
        color='region'
    )
    st.plotly_chart(fig, use_container_width=True)
```

**2. Add new filter:**
```python
# In sidebar section
new_filter = st.sidebar.multiselect(
    "Select New Filter:",
    options=df['new_column'].unique(),
    default=df['new_column'].unique()[:5]
)

# Apply filter
filtered_df = filtered_df[
    filtered_df['new_column'].isin(new_filter)
]
```

**3. Deploy update:**
```bash
git add app.py
git commit -m "Add new feature to dashboard"
git push origin main
# Streamlit Cloud auto-redeploys in ~2 minutes
```

---

## 📚 Resources

**Streamlit Documentation:**
- Official docs: https://docs.streamlit.io/
- Gallery (examples): https://streamlit.io/gallery
- Forum (help): https://discuss.streamlit.io/

**Plotly Documentation:**
- Plotly Express: https://plotly.com/python/plotly-express/
- Figure reference: https://plotly.com/python/reference/

**Tutorials:**
- Streamlit basics: https://docs.streamlit.io/get-started
- Plotly tutorial: https://plotly.com/python/getting-started/

---

## 🤝 Sharing and Citation

### In Academic Work

**In methodology section:**
```
Data exploration was facilitated through an interactive Streamlit dashboard
(available at https://[your-url].streamlit.app), allowing dynamic filtering
by country, article, time period, and case characteristics.
```

**In figures:**
```
Figure 1: Violation rates by country (2000-2024)
Source: ECHR Case Explorer dashboard, filtered to Eastern Europe
```

### On Social Media

**Example tweet:**
```
🎓 New research tool: Interactive dashboard for 1,904 ECHR cases (2000-2024)

⚖️ Filter by country, article, region
📊 Dynamic visualizations
📥 Download filtered data

Explore: https://[your-url].streamlit.app
GitHub: https://github.com/mattokumus/assignment2

#ECHR #HumanRights #DataScience
```

---

## ❓ FAQ

**Q: Can I use this dashboard in my thesis/dissertation?**
A: Yes! Include the URL and cite the underlying dataset. Take screenshots for appendices.

**Q: Is Streamlit Cloud really free?**
A: Yes, for public GitHub repositories. Free tier includes: unlimited viewers, auto-deployment, SSL certificate.

**Q: Can I password-protect the dashboard?**
A: Not on free tier. Use Streamlit for Teams (paid) or add custom authentication.

**Q: How do I update the data?**
A: Re-run `python3 assignment2.py` to regenerate `extracted_data.csv`, then restart dashboard or redeploy.

**Q: Can I change the design/colors?**
A: Yes, edit `app.py`. See "Configuration Options" section above.

**Q: Will the dashboard work with more data?**
A: Yes, Streamlit handles millions of rows. May need performance tuning (see "Performance Tuning" section).

**Q: Can I add my own analyses?**
A: Absolutely! Add tabs/sections in `app.py`. See "Adding New Features" above.

---

## 📧 Support

**Dashboard issues:** Open GitHub issue at https://github.com/mattokumus/assignment2/issues

**Streamlit questions:** https://discuss.streamlit.io/

**ECHR data questions:** See DATA_PROVENANCE.md

---

**Document Version:** 1.0
**Last Updated:** November 2024
**Maintainer:** mattokumus
**Dashboard URL:** [Add after deployment]

---

**🎉 Enjoy exploring the data!**
