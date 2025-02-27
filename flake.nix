{
  description = "Double O Seven's flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachSystem 
    [ "aarch64-darwin" ] 
    (system: 
      let 
        pkgs = nixpkgs.legacyPackages.${system};
        pythonWithPkgs = pkgs.python3.withPackages (ps: with ps; [
          langchain
          langchain-ollama
          langchain-openai
          langchain-community
          langchain-text-splitters
          langgraph
          pypdf
          requests
          numpy
          transformers
          torch
          beautifulsoup4
          ipython
        ]);
      in
    {
      devShells.default = with pkgs; mkShell {
          name = "007";
          buildInputs = [
            pythonWithPkgs 
            pyright
          ];
      };
    });
}
