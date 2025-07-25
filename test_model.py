import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test input
news = """
BRUSSELS (Reuters) - A leading European rights watchdog called on Turkey on Friday to ease post-coup state of emergency laws that have seen thousands arrested and restore power to regional authorities. President Tayyip Erdogan has overseen a mass purge in the armed forces and the judiciary, as well as a crackdown on critics including academics and journalists since a failed military coup in July last year.  An advisory body to the Council of Europe, of which Turkey is a member, acknowledged in a report  the need for certain extraordinary steps taken by Turkish authorities to face a dangerous armed conspiracy .  However...Turkish authorities have interpreted these extraordinary powers too extensively,  said the experts, known as the Venice Commission, in an opinion that has no legal force. It urged Ankara to lift laws allowing it to pick mayors, deputy mayors and members of local councils outside of local elections, a reference to rules the Turkish government has used to replace local pro-Kurdish politicians across the largely Kurdish southeast of the country. The experts at the Council of Europe, Europe s leading human rights organization with 47 member states, recommended Turkey set a time limit on the emergency rules and ensure proper judicial oversight of any counter-terrorism measures. Erdogan has accused U.S.-based, Islamic cleric Fethullah Gulen of orchestrating the coup, in which 250 people were killed. Gulen has denied involvement.  Since then, more than 50,000 people have been jailed pending trial over links to Gulen, while 150,000 people have been sacked or suspended from jobs in the public and private sectors. Rights groups and Turkey s Western allies - including the European Union - have been taken aback by the scale of the purge, sounding alarm that Erdogan was using the coup as a pretext to quash dissent. The government in Turkey, a NATO member, says it must neutralize the threat represented by Gulen s network deeply rooted in the army, schools and courts.
"""

# Predict
news_vector = vectorizer.transform([news])
prediction = model.predict(news_vector)[0]
print("Prediction:", "Real News 🟢" if prediction == 1 else "Fake News 🔴")
