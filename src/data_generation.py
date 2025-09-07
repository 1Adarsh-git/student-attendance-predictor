import pandas as pd
import numpy as np
import random
from itertools import combinations

np.random.seed(42)
random.seed(42)

def generate_users_data(n_users=1000):
    """Generate users.csv with synthetic user data"""

    departments = ['CSE', 'ECE', 'ME', 'CE', 'EE', 'IT']
    possible_interests = ['ML', 'AI', 'Web', 'Finance', 'Marketing', 'Robotics', 'Design', 'Product']

    users_data = []

    for i in range(1, n_users + 1):
        user_id = f'u{i}'
        dept = np.random.choice(departments)
        year = np.random.randint(1, 5)  # 1-4 years

        n_interests = np.random.randint(1, 5)
        interests = np.random.choice(possible_interests, size=n_interests, replace=False)
        interests_str = ','.join(interests)

        past_attendance_count = np.random.poisson(4)  
        past_attendance_count = min(past_attendance_count, 20)  

        users_data.append({
            'user_id': user_id,
            'dept': dept, 
            'year': year,
            'interests': interests_str,
            'past_attendance_count': past_attendance_count
        })

    return pd.DataFrame(users_data)

def generate_events_data(n_events=200):
    """Generate events.csv with synthetic event data"""

    event_types = ['Workshop', 'Talk', 'Hackathon', 'Competition']
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    times_of_day = ['Morning', 'Afternoon', 'Evening']
    possible_tags = ['ML', 'AI', 'Web', 'Finance', 'Marketing', 'Robotics', 'Design', 'Product']

    
    title_templates = {
        'Workshop': ['Intro to {}', '{} Workshop', 'Hands-on {}', '{} Bootcamp'],
        'Talk': ['{} Talk', 'Understanding {}', '{} Insights', 'Future of {}'],
        'Hackathon': ['{} Hackathon', '{} Challenge', 'Build with {}'],
        'Competition': ['{} Competition', '{} Contest', '{} Challenge']
    }

    events_data = []

    for i in range(1, n_events + 1):
        event_id = f'e{i}'
        event_type = np.random.choice(event_types)

        
        n_tags = np.random.randint(1, 4)
        tags = np.random.choice(possible_tags, size=n_tags, replace=False)
        tags_str = ','.join(tags)

        
        template = np.random.choice(title_templates[event_type])
        main_tag = tags[0] if len(tags) > 0 else 'Tech'
        title = template.format(main_tag)

        
        description = f"Learn about {', '.join(tags)} in this {event_type.lower()}"

        day_of_week = np.random.choice(days_of_week)
        time_of_day = np.random.choice(times_of_day)

        events_data.append({
            'event_id': event_id,
            'title': title,
            'description': description,
            'tags': tags_str,
            'event_type': event_type,
            'day_of_week': day_of_week,
            'time_of_day': time_of_day
        })

    return pd.DataFrame(events_data)

def calculate_tag_match_score(user_interests, event_tags):
    """Calculates tags matching between user interests and event tags"""
    user_tags = set(user_interests.split(','))
    event_tag_set = set(event_tags.split(','))
    return len(user_tags.intersection(event_tag_set))

def generate_attendance_data(users_df, events_df, n_samples=5000):
    """Generate attendance.csv with realistic attendance patterns"""

    register_channels = ['email', 'whatsapp', 'instagram', 'none']

    attendance_data = []

    for _ in range(n_samples):
        
        user = users_df.sample(1).iloc[0]
        event = events_df.sample(1).iloc[0]

        
        tag_match_score = calculate_tag_match_score(user['interests'], event['tags'])
        notification_received = np.random.binomial(1, 0.7)  
        distance_km = np.random.uniform(0, 30)
        register_channel = np.random.choice(register_channels)

        p = 0.05  

        
        if notification_received:
            p += 0.20

        
        p += tag_match_score * 0.12

        
        if user['past_attendance_count'] > 5:
            p += 0.10

        
        if event['time_of_day'] == 'Evening' and user['year'] >= 2:
            p += 0.05

        
        p -= 0.01 * distance_km

        
        if event['event_type'] in ['Workshop', 'Hackathon']:
            p += 0.05
        elif event['event_type'] == 'Talk':
            p += 0.01

        
        p = max(0, min(1, p))

        
        attend = np.random.binomial(1, p)

        attendance_data.append({
            'user_id': user['user_id'],
            'event_id': event['event_id'], 
            'dept': user['dept'],
            'year': user['year'],
            'interests': user['interests'],
            'past_attendance_count': user['past_attendance_count'],
            'event_type': event['event_type'],
            'event_tags': event['tags'],
            'day_of_week': event['day_of_week'],
            'time_of_day': event['time_of_day'],
            'notification_received': notification_received,
            'distance_km': distance_km,
            'register_channel': register_channel,
            'attend': attend
        })

    return pd.DataFrame(attendance_data)

def main():
    print("Generating synthetic datasets...")

    
    print(" Generating users.csv...")
    users_df = generate_users_data(1000)
    users_df.to_csv('data/users.csv', index=False)
    print(f"  Created users.csv with {len(users_df)} records")

    
    print(" Generating events.csv...")
    events_df = generate_events_data(200) 
    events_df.to_csv('data/events.csv', index=False)
    print(f"  Created events.csv with {len(events_df)} records")


    print(" Generating attendance.csv...")
    attendance_df = generate_attendance_data(users_df, events_df, 5000)
    attendance_df.to_csv('data/attendance.csv', index=False)
    print(f"  Created attendance.csv with {len(attendance_df)} records")


    attendance_rate = attendance_df['attend'].mean()
    print(f"\nDataset Statistics:")
    print(f" Overall attendance rate: {attendance_rate:.2%}")
    print(f" Positive class samples: {attendance_df['attend'].sum()}")
    print(f" Negative class samples: {len(attendance_df) - attendance_df['attend'].sum()}")

    print("\nSample records:")
    print("\nUsers sample:")
    print(users_df.head(3).to_string())
    print("\nEvents sample:")
    print(events_df.head(3).to_string())
    print("\nAttendance sample:")
    print(attendance_df.head(3).to_string())

if __name__ == "__main__":
    main()
