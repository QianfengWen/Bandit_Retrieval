from abc import ABC, abstractmethod

from .datasets import (
    RestaurantPhiDataset,
    RestaurantNorDataset,
    TravelDestDataset,
    PointRecUSDataset,
    HotelChicagoDataset,
    HotelLondonDataset,
    HotelMontrealDataset,
    HotelNYCDataset,
)


def handle_dataset(dataset_name):
    if dataset_name == "restaurant_phi":
        dataset = RestaurantPhiDataset()
    elif dataset_name == "restaurant_nor":
        dataset = RestaurantNorDataset()
    elif dataset_name == "travel_dest":
        dataset = TravelDestDataset()
    elif dataset_name == "point_rec_us":
        dataset = PointRecUSDataset()
    elif dataset_name == "hotel_chicago":
        dataset = HotelChicagoDataset()
    elif dataset_name == "hotel_london":
        dataset = HotelLondonDataset()
    elif dataset_name == "hotel_montreal":
        dataset = HotelMontrealDataset()
    elif dataset_name == "hotel_nyc":
        dataset = HotelNYCDataset()
    return dataset

class Dataloader(ABC):    
    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_questions(self):
        pass

    @abstractmethod
    def load_passages(self):
        pass

    @abstractmethod
    def create_relevance_map(self):
        pass

    def load_data(self):
        self.dataset = self.load_dataset()
        question_ids, question_texts = self.load_questions()
        passage_ids, passage_texts = self.load_passages()
        relevance_map = self.create_relevance_map()
        
        return question_ids, question_texts, passage_ids, passage_texts, relevance_map

    def visualize_relevance(self, fig_path):
        relevance = [qrel.relevance for qrel in self.dataset.qrels_iter()]
        plt.hist(relevance, bins=len(set(relevance)))
        plt.xlabel('Relevance')
        plt.ylabel('Frequency')
        plt.title('Relevance Distribution')
        plt.savefig(fig_path)
        return
    
    def save_relevance_map(self, data_dir, relevance_map):
        data_path = os.path.join(data_dir, "relevance_map.json")
        with open(data_path, 'w') as json_file:
            json.dump(relevance_map, json_file, indent=4)
        return
    



        
