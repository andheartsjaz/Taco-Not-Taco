# grab magic ml library that literally does everything
import turicreate as tc

# load images
data = tc.image_analysis.load_images(
    '../images',
    with_path=True
)

# label data as taco or nottaco based on filepath
data['label'] = data['path'].apply(
    lambda path: 'taco' if '/tacos/' in path else 'nottacos'
)

# split data into test and train
train_data, test_data = data.random_split(0.8)

# literally train everything based on the pre-existing squeezenet model
# and training data from above
model = tc.image_classifier.create(
    train_data,
    target='label',
    model='squeezenet_v1.1',
    max_iterations=50
)

# test model
model.evaluate(test_data)

# save model away 
model.export_coreml('taco.mlmodel')
