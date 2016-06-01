package CoreferenceAlliance;


public class App
{
   public static void main( String[] args )
   {
       LSTM lstm = new LSTM();
       
       lstm.TrainFromXMLDir("C:\\Users\\michael\\Documents\\GitHub\\CoreferenceNN\\DATA\\BIONLP\\Michael_XML\\PMID-1313226.xml");
   }
}