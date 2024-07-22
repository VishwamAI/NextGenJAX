import unittest
import os
import shutil
from nextgenjax.transformer_models import TransformerModel

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        # Initialize with a known small model for testing
        self.model_name = "sshleifer/tiny-gpt2"
        self.transformer_model = TransformerModel(model_name=self.model_name)
        self.save_directory = "/tmp/test_transformer_model_save"

    def tearDown(self):
        # Clean up the test directory after each test
        if os.path.exists(self.save_directory):
            shutil.rmtree(self.save_directory)

    def test_init(self):
        # Test initialization
        self.assertEqual(self.transformer_model.model_name, self.model_name)
        self.assertIsNotNone(self.transformer_model.model)
        self.assertIsNotNone(self.transformer_model.tokenizer)

    def test_save_model(self):
        # Test saving the model and tokenizer
        self.transformer_model.save_model(save_directory=self.save_directory)
        # Check if the files have been created
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "tokenizer")))
        self.assertTrue(os.path.exists(os.path.join(self.save_directory, "model")))

    def test_load_model(self):
        # First, save the model to ensure we have something to load
        self.test_save_model()

        # Create a new instance to load the saved model into
        new_model = TransformerModel(model_name=self.model_name)
        new_model.load_model(load_directory=self.save_directory)

        # Check if the model and tokenizer are loaded correctly
        self.assertIsNotNone(new_model.model)
        self.assertIsNotNone(new_model.tokenizer)

        # Test if the loaded model can generate text
        input_text = "This is a test sentence."
        generated_text = new_model.generate_text(input_text=input_text)
        self.assertIsInstance(generated_text, str)
        self.assertNotEqual(generated_text, "")

    def test_generate_text(self):
        # Test text generation
        input_text = "This is a test sentence."
        generated_text = self.transformer_model.generate_text(input_text=input_text)
        self.assertIsInstance(generated_text, str)
        self.assertNotEqual(generated_text, "")

        # Test with invalid input_text type
        with self.assertRaises(ValueError):
            self.transformer_model.generate_text(input_text=123)

        # Test with invalid max_length type
        with self.assertRaises(ValueError):
            self.transformer_model.generate_text(input_text=input_text, max_length="invalid")

        # Test with invalid max_length value
        with self.assertRaises(ValueError):
            self.transformer_model.generate_text(input_text=input_text, max_length=0)

if __name__ == '__main__':
    unittest.main()
