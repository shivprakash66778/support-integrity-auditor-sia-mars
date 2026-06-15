# Dataset Instructions

## Required Dataset

**Customer Support Tickets — CRM Dataset**

Download from Kaggle:
https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data

## Setup

1. Download `customer_support_tickets.csv` from the link above
2. Place it in this folder: `data/customer_support_tickets.csv`

## Required Columns

| Column | Role |
|--------|------|
| Ticket_ID | Unique identifier |
| Ticket_Subject | Short summary |
| Ticket_Description | Full problem text |
| Issue_Category | Category of issue |
| Priority_Level | Human-assigned priority (Low/Medium/High/Critical) |
| Ticket_Channel | Intake channel (Email, Chat, etc.) |
| Resolution_Time_Hours | Time to resolve |
| Satisfaction_Score | Customer satisfaction (1-5) |

## Sample

A `sample_input.csv` with 20 rows is included for testing the prediction pipeline
without the full dataset.
