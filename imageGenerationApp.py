import streamlit as st

'''
{focusDetailed}: A serene mountain landscape at sunrise
{adjective1}: Tranquil
{adjective2}: Luminous
{visualStyle1}: Impressionistic
{visualStyle2}: Vivid color palette
{visualStyle3}: Soft brushstrokes
{artistReference}: Claude Monet

A serene mountain landscape at sunrise, 
tranquil and luminous, 
rendered in an impressionistic style with a vivid color palette and soft brushstrokes, 
reminiscent of the works of Claude Monet.
'''

link = "https://image.pollinations.ai/prompt/"

st.title("Image Generation App")


query = st.text_input("Enter the query")

col1, col2, col3 = st.columns(3)

with col1:
    adj1 = st.text_input("Adjective 1")
    vs2 = st.text_input("Visual Style 2")
with col2:
    adj2 = st.text_input("Adjective 2")
    vs3 = st.text_input("Visual Style 3")
with col3:
    vs1 = st.text_input("Visual Style 1")
    day = st.text_input("Artist Reference")

if query and adj1 and adj2 and vs1 and vs2 and vs3 and day: 
    final = '%20'.join(query.split()) + ",%20" + '%20'.join(adj1.split()) + ",%20" + '%20'.join(adj2.split()) + \
    ",%20" + '%20'.join(vs1.split()) + ",%20" + '%20'.join(vs2.split()) + ",%20" + '%20'.join(vs3.split()) + ",%20" + '%20'.join(day.split()) 
    
btn = st.button("Generate")
    
if btn:
    st.image(link + final)
    #final = '%20'.join(query.split()) + ",%20" + '%20'.join(adj1.split()) + ",%20" + '%20'.join(adj2.split()) + ",%20" + '%20'.join(day.split()) 
    #st.image(link + final)
